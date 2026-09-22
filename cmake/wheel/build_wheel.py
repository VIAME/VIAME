#!/usr/bin/env python3
"""Build a PEP 427 wheel from a VIAME install prefix.

The wheel's contents are *data*, not code: `contents.txt` beside this file
says which paths of the install prefix go in and where they land. That is the
whole reason this is written the way it is -- `main` installs a `kwiver`
package beside `viame` and 65 separate shared libraries, `lite` installs one
`viame` package and a single `libviame.so`, and the two branches have to share
this script so that merging one into the other is a change to a list rather
than a conflict in a builder.

Native libraries are not relocated. VIAME's extension modules already carry a
`$ORIGIN`-relative RUNPATH --

    $ORIGIN/../../../../../lib

-- which, from `site-packages/<pkg>/<sub>/x.so`, resolves to the environment's
`lib/`. A wheel can put a file exactly there through the `.data/data/` scheme,
so the libraries go to `<name>-<version>.data/data/lib/` and the RUNPATH that
worked in the install prefix keeps working in the installed wheel. See
`docs/wheels.md` for the layouts where that does not hold.

What *is* patched is the CUDA search path: RUNPATH entries are appended so
`libcudart` and friends come from the `nvidia/*` pip wheels torch installs
rather than from a system `/usr/local/cuda`. That needs `patchelf`, and
degrades to a note rather than a failure without it. See `add_cuda_rpath`.
"""

import argparse
import base64
import csv
import hashlib
import io
import posixpath
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


# ----------------------------------------------------------------------------
# The contents file
#
#   include <glob> -> <destination>
#   exclude <glob>
#
# Globs are relative to the install prefix and matched against POSIX paths.
# `**` crosses directory separators; a trailing `/` on the destination means
# "a directory, keep the part of the path the glob's own directory prefix did
# not consume". Excludes are applied after includes and win, so a package can
# be taken whole and then thinned.
# ----------------------------------------------------------------------------

class Rule:
    __slots__ = ("action", "pattern", "dest", "lineno")

    def __init__(self, action, pattern, dest, lineno):
        self.action = action
        self.pattern = pattern
        self.dest = dest
        self.lineno = lineno


def read_contents(path):
    rules = []
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "->" in line:
            head, dest = (p.strip() for p in line.split("->", 1))
        else:
            head, dest = line, None
        parts = head.split(None, 1)
        if len(parts) != 2:
            raise SystemExit(f"{path}:{lineno}: expected '<action> <glob>', got {raw!r}")
        action, pattern = parts[0], parts[1].strip()
        if action not in ("include", "exclude"):
            raise SystemExit(f"{path}:{lineno}: unknown action {action!r}")
        if action == "include" and not dest:
            raise SystemExit(f"{path}:{lineno}: an include needs '-> <destination>'")
        rules.append(Rule(action, pattern, dest, lineno))
    if not rules:
        raise SystemExit(f"{path}: no rules")
    return rules


def _translate(pattern):
    """Regex for a glob, capturing what the last `**` matches.

    The capture is what makes a destination directory work: `include
    lib/python3.*/site-packages/viame/** -> viame/` has to mean "whatever
    `**` matched is the path under `viame/`". Deriving that from the glob's
    leading literal directories instead -- which is what this did first --
    stops at the first wildcard, so `python3.*` left `python3.10/site-packages`
    on the front of every destination and the wheel installed a `kwiver`
    package containing a `kwiver` package.
    """
    out, i = [], 0
    stars = [j for j in range(len(pattern)) if pattern.startswith("**", j)]
    last = stars[-1] if stars else -1
    while i < len(pattern):
        if i == last:
            if pattern.startswith("**/", i):
                out.append("(?:(.*)/)?")
                i += 3
            else:
                out.append("(.*)")
                i += 2
        elif pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z"), last >= 0


def _match(pattern, rel):
    """True if `rel` matches `pattern`."""
    return _translate(pattern)[0].match(rel) is not None


def _match_tail(pattern, rel):
    """(matched, tail) -- `tail` is what the last `**` consumed, else None."""
    rx, has_star = _translate(pattern)
    m = rx.match(rel)
    if not m:
        return False, None
    if not has_star:
        return True, None
    return True, (m.group(1) or "")


def _static_prefix(pattern):
    """The leading directories of a glob that contain no wildcard.

    Only used to avoid walking the whole prefix; it is not what decides a
    destination path any more. See `_translate`.
    """
    parts = []
    for part in pattern.split("/"):
        if any(ch in part for ch in "*?["):
            break
        parts.append(part)
    return "/".join(parts)


def read_manifest(path):
    """The set of files a build installed, from CMake's `install_manifest.txt`.

    An install prefix is an accumulation: `make install` only ever adds, so a
    prefix that has been built into for months holds files whose sources were
    deleted long ago. Selecting a wheel's contents by walking it therefore
    packs other builds' leavings. On this branch that was twelve files --
    two `.bak_prereid` backups, `video_io/image_viewer.py` and
    `video_io/pil_image_io.py` at the path they had before the `image_io`
    split moved them, and a `viame/home/local/.../core.py` tree left by a
    build that resolved an absolute path as a relative one.

    None of them would have broken the wheel, which is what makes this worth
    a filter rather than a cleanup: a stale module that still imports is how
    a package ends up shipping code nobody can find in the tree.
    """
    manifest = Path(path)
    if not manifest.is_file():
        # The manifest only exists after an install, and a superbuild may put
        # it somewhere other than the top of the build tree. Missing is not
        # fatal: the filter is an improvement on walking the prefix, not a
        # precondition for building a wheel. Said loudly, because without it
        # the wheel can carry files from builds whose sources are gone.
        print(f"  note: {path} does not exist, so the wheel is selected by "
              f"walking the prefix. Anything an earlier build left there and "
              f"the contents file matches will be packed.", file=sys.stderr)
        return None

    files = set()
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line:
            files.add(str(Path(line).resolve()))
    if not files:
        raise SystemExit(f"{path}: empty manifest")
    return files


def select(prefix, rules, data_dir, scripts_dir=None, manifest=None):
    """Return {wheel-relative path: source Path}.

    Two placeholders, being the two install schemes this needs.

    `{data}` is `<name>-<version>.data/data`, which pip unpacks into the
    environment prefix. `{data}/lib/` is therefore `<env>/lib`, the one place
    a wheel can put a shared library where an `$ORIGIN/../../../../../lib`
    RUNPATH will find it.

    `{scripts}` is `<name>-<version>.data/scripts`, unpacked into `<env>/bin`
    and marked executable. It is deliberately **not** spelled
    `{data}/scripts`: that is `.data/data/scripts`, which installs to
    `<env>/scripts`, a directory nothing looks in. It packs, the mode bits
    are right, and the command is simply not on the PATH.
    """
    prefix = Path(prefix)
    if not prefix.is_dir():
        raise SystemExit(f"{prefix}: not a directory")

    chosen = {}
    skipped = []
    for rule in (r for r in rules if r.action == "include"):
        base = _static_prefix(rule.pattern)
        root = prefix / base if base else prefix
        if not root.exists():
            print(f"  note: nothing at {base or '.'} for line {rule.lineno}", file=sys.stderr)
            continue
        walk = root.rglob("*") if root.is_dir() else [root]
        hits = 0
        for src in walk:
            if not src.is_file() or src.is_symlink():
                continue
            rel = src.relative_to(prefix).as_posix()
            matched, tail = _match_tail(rule.pattern, rel)
            if not matched:
                continue
            if manifest is not None and str(src.resolve()) not in manifest:
                skipped.append(rel)
                continue
            rule_dest = rule.dest.replace("{data}", data_dir)
            if scripts_dir is not None:
                rule_dest = rule_dest.replace("{scripts}", scripts_dir)
            if rule_dest.endswith("/"):
                # No `**` in the pattern means the glob names files in one
                # directory, so the file's own name is the tail. A shared
                # library goes under its SONAME, which is what its dependants
                # ask the loader for; see `soname`.
                if tail is None:
                    so = soname(src) if ".so" in src.name else None
                    dest = rule_dest + (so or src.name)
                else:
                    dest = rule_dest + tail
            else:
                dest = rule_dest
            chosen[dest] = src
            hits += 1
        if not hits:
            print(f"  note: line {rule.lineno} matched nothing: {rule.pattern}", file=sys.stderr)

    for rule in (r for r in rules if r.action == "exclude"):
        for dest in [d for d, s in chosen.items()
                     if _match(rule.pattern, s.relative_to(prefix).as_posix())]:
            del chosen[dest]

    # Report only what an exclude rule would not have dropped anyway. Most of
    # what a prefix holds beyond its manifest is `__pycache__`, and burying
    # the interesting entries under three hundred of those is the same as not
    # reporting them.
    notable = [rel for rel in skipped
               if not any(_match(r.pattern, rel)
                          for r in rules if r.action == "exclude")]
    if notable:
        print(f"  {len(notable)} file(s) in the prefix but not in this build's "
              f"install manifest, left out:", file=sys.stderr)
        for rel in sorted(notable)[:12]:
            print(f"    {rel}", file=sys.stderr)
        if len(notable) > 12:
            print(f"    ... and {len(notable) - 12} more", file=sys.stderr)

    if not chosen:
        raise SystemExit("the contents file selected no files")
    return chosen


# ----------------------------------------------------------------------------
# SONAME
#
# A shared library has to be packed under the name its dependants ask for,
# which is its SONAME and not necessarily its filename. In an install prefix
# the two are bridged by a symlink -- `libviame.so.1 -> libviame.so.1.0.0` --
# and a wheel cannot rely on carrying symlinks, so the real file is packed
# under the SONAME instead. Packing three copies under all three names would
# also work and would cost 105 MB for one 52 MB library.
#
# `main` does not need this and `lite` does: there `libvital_types.so.2.4.1`
# is its own SONAME, here `libviame.so.1.0.0`'s is `libviame.so.1`. The first
# version of this script skipped symlinks and packed real files under their
# own names, which is why the lite wheel built, installed, and failed at
# import with `libviame.so.1: cannot open shared object file`.
#
# Read here rather than shelled out to `objdump` so that building a wheel
# needs nothing but python.
# ----------------------------------------------------------------------------

def soname(path):
    """The ELF SONAME of `path`, or None if it has none or is not an ELF."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return None
    if data[4] != 2:            # 64-bit only; VIAME ships no 32-bit library
        return None
    endian = "<" if data[5] == 1 else ">"
    u16 = lambda o: struct.unpack_from(endian + "H", data, o)[0]
    u64 = lambda o: struct.unpack_from(endian + "Q", data, o)[0]

    e_phoff, e_phentsize, e_phnum = u64(0x20), u16(0x36), u16(0x38)
    loads, dynamic = [], None
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        if off + 56 > len(data):
            return None
        p_type = struct.unpack_from(endian + "I", data, off)[0]
        p_offset, p_vaddr = u64(off + 0x08), u64(off + 0x10)
        p_filesz = u64(off + 0x20)
        if p_type == 1:                       # PT_LOAD
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == 2:                     # PT_DYNAMIC
            dynamic = (p_offset, p_filesz)
    if not dynamic:
        return None

    def to_offset(vaddr):
        for v, o, sz in loads:
            if v <= vaddr < v + sz:
                return o + (vaddr - v)
        return None

    d_off, d_size = dynamic
    strtab = name_off = None
    for i in range(d_size // 16):
        o = d_off + i * 16
        if o + 16 > len(data):
            break
        tag, val = u64(o), u64(o + 8)
        if tag == 0:                          # DT_NULL
            break
        if tag == 5:                          # DT_STRTAB
            strtab = val
        elif tag == 14:                       # DT_SONAME
            name_off = val
    if strtab is None or name_off is None:
        return None
    base = to_offset(strtab)
    if base is None:
        return None
    end = data.find(b"\0", base + name_off)
    if end < 0:
        return None
    return data[base + name_off:end].decode("ascii", "replace") or None


# ----------------------------------------------------------------------------
# Stripping
#
# Symbol tables are ~40% of what this installs and there is no DWARF behind
# them. Stripped into a scratch copy, never in place: people run from the
# install prefix.
# ----------------------------------------------------------------------------

def strip_into(src, scratch, seen):
    """A stripped copy of `src` under `scratch`, or `src` if stripping fails.

    Returns the path to pack. `seen` maps an already-stripped source to its
    copy so that a library packed under two names is only stripped once.
    """
    if src in seen:
        return seen[src]
    try:
        with open(src, "rb") as f:
            if f.read(4) != b"\x7fELF":
                seen[src] = src
                return src
    except OSError:
        seen[src] = src
        return src

    dst = scratch / f"{len(seen)}-{src.name}"
    try:
        shutil.copy2(src, dst)
        r = subprocess.run(["strip", "--strip-all", str(dst)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            # A library that will not strip is packed as it is rather than
            # failing the build; the size target is not worth a wheel that
            # cannot be produced on a machine without binutils.
            print(f"  note: strip failed for {src.name}: {r.stderr.strip()[:120]}",
                  file=sys.stderr)
            dst.unlink(missing_ok=True)
            seen[src] = src
            return src
    except (OSError, FileNotFoundError) as e:
        print(f"  note: cannot strip ({e}); packing unstripped", file=sys.stderr)
        seen[src] = src
        return src
    seen[src] = dst
    return dst


# ----------------------------------------------------------------------------
# CUDA from the pip wheels
#
# The binaries link `libcudart` and friends with no path for them, so a fresh
# environment finds whatever `/usr/local/cuda` holds, or nothing. torch ships
# those libraries as wheels; RUNPATH entries here point at them.
#
# Done when packing rather than at link time because the relative path
# differs per destination, and only the packer knows where a file lands.
# Existing entries are kept so `libviame` still resolves, and a component
# whose wheel is absent falls through to the system copy.
#
# Requires `patchelf` (on PyPI); degrades to a note without it.
# ----------------------------------------------------------------------------

# Where the nvidia wheels put their libraries, which differs by CUDA major:
# cu12 uses one directory per component, cu13 consolidated the core runtime
# into `nvidia/cu13/lib` and dropped the suffix on those distributions.
# Both sets are listed for either variant -- an entry that does not exist
# costs nothing, and it keeps resolving when torch and VIAME disagree.
CUDA_WHEEL_DIRS = {
    12: (
        "nvidia/cuda_runtime/lib",   # libcudart
        "nvidia/cublas/lib",         # libcublas, libcublasLt
        "nvidia/curand/lib",         # libcurand
        "nvidia/cudnn/lib",          # libcudnn and its engines
        "nvidia/cuda_nvrtc/lib",     # libnvrtc, via cudnn's JIT path
    ),
    13: (
        "nvidia/cu13/lib",           # libcudart, libcublas, libcurand, nvrtc
        "nvidia/cudnn/lib",          # still its own, still suffixed
    ),
}


def _env_relative(dest, dist, purelib):
    """Where a wheel entry lands under the environment prefix."""
    scripts = f"{dist}.data/scripts/"
    data = f"{dist}.data/data/"
    if dest.startswith(scripts):
        return "bin/" + dest[len(scripts):]
    if dest.startswith(data):
        return dest[len(data):]
    return f"{purelib}/{dest}"


def cuda_rpath_entries(dest, dist, purelib, major=None):
    """`$ORIGIN`-relative RUNPATH entries reaching the nvidia wheels."""
    here = posixpath.dirname(_env_relative(dest, dist, purelib))
    components = []
    for dirs in ([CUDA_WHEEL_DIRS[major]] if major in CUDA_WHEEL_DIRS
                 else CUDA_WHEEL_DIRS.values()):
        for c in dirs:
            if c not in components:
                components.append(c)
    return [f"$ORIGIN/{posixpath.relpath(f'{purelib}/{c}', here or '.')}"
            for c in components]


def add_cuda_rpath(path, dest, dist, purelib, state, major=None):
    """Append the nvidia wheel directories to `path`'s RUNPATH, in place."""
    if state["patchelf"] is None:
        state["patchelf"] = shutil.which("patchelf") or False
    if not state["patchelf"]:
        state["skipped"] += 1
        return False
    try:
        current = subprocess.run([state["patchelf"], "--print-rpath", str(path)],
                                 capture_output=True, text=True)
        if current.returncode != 0:
            return False          # not a dynamic executable; nothing to do
        existing = [e for e in current.stdout.strip().split(":") if e]
        wanted = [e for e in cuda_rpath_entries(dest, dist, purelib, major)
                  if e not in existing]
        if not wanted:
            return False
        r = subprocess.run(
            [state["patchelf"], "--set-rpath", ":".join(existing + wanted), str(path)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  note: patchelf failed on {dest}: {r.stderr.strip()[:120]}",
                  file=sys.stderr)
            return False
    except OSError as e:
        print(f"  note: patchelf unusable ({e})", file=sys.stderr)
        state["patchelf"] = False
        return False
    state["patched"] += 1
    return True


# ----------------------------------------------------------------------------
# Wheel metadata
# ----------------------------------------------------------------------------

def _hash(path):
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            size += len(block)
    digest = base64.urlsafe_b64encode(h.digest()).rstrip(b"=").decode()
    return f"sha256={digest}", size


def metadata(name, version, summary, requires, description, requires_python):
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        f"Summary: {summary}",
        "License: BSD-3-Clause",
        f"Requires-Python: >={requires_python}",
    ]
    lines += [f"Requires-Dist: {r}" for r in requires]
    return "\n".join(lines) + "\n\n" + description + "\n"


def wheel_metadata(root_is_purelib, tag):
    return (
        "Wheel-Version: 1.0\n"
        "Generator: viame cmake/wheel/build_wheel.py\n"
        f"Root-Is-Purelib: {'true' if root_is_purelib else 'false'}\n"
        f"Tag: {tag}\n"
    )


def build(args):
    rules = [r for f in args.contents for r in read_contents(f)]
    version = (f"{args.version}+{args.local_version}"
               if args.local_version else args.version)
    dist = f"{args.name}-{version}"
    manifest = read_manifest(args.manifest) if args.manifest else None
    chosen = select(args.prefix, rules, f"{dist}.data/data",
                    f"{dist}.data/scripts", manifest)

    tag = f"{args.python_tag}-{args.abi_tag}-{args.platform_tag}"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    whl = out / f"{dist}-{tag}.whl"

    requires = [r for r in (args.requires or []) if r]
    for path in (args.requires_from or []):
        for line in Path(path).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                requires.append(line)
    if not requires:
        print("  note: the wheel declares no dependencies", file=sys.stderr)
    records = []
    staged = {}
    raw = packed = 0
    scratch = Path(tempfile.mkdtemp(prefix="viame-wheel-"))
    cuda = {"patchelf": None, "patched": 0, "skipped": 0}
    # Where a purelib entry lands under the environment prefix, which is what
    # the `$ORIGIN`-relative CUDA paths are measured from. Derived from the
    # wheel's own python tag, so it agrees with the interpreter the extension
    # modules were built against by construction.
    purelib = f"lib/python{args.python_tag[2]}.{args.python_tag[3:]}/site-packages"

    try:
        with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
            for dest in sorted(chosen):
                src = chosen[dest]
                raw += src.stat().st_size
                # A scratch copy whenever the bytes are to be modified. The
                # install prefix is never written to: people run from it.
                if (args.strip or args.cuda_from_wheels) and (src, dest) not in staged:
                    work = strip_into(src, scratch, {}) if args.strip else src
                    if args.cuda_from_wheels:
                        if work is src:          # stripping declined to copy
                            work = scratch / f"{len(staged)}-{src.name}"
                            shutil.copy2(src, work)
                        add_cuda_rpath(work, dest, dist, purelib, cuda,
                                       args.cuda_major)
                    staged[(src, dest)] = work
                src = staged.get((src, dest), src)
                packed += src.stat().st_size
                z.write(src, dest)
                digest, size = _hash(src)
                records.append((dest, digest, size))

            info = f"{dist}.dist-info"
            extras = {
                f"{info}/METADATA": metadata(args.name, version, args.summary,
                                             requires, args.description,
                                             args.requires_python),
                f"{info}/WHEEL": wheel_metadata(False, tag),
                f"{info}/top_level.txt": "".join(
                    n + "\n" for n in sorted(args.top_level or [])),
            }
            if args.entry_points and Path(args.entry_points).is_file():
                extras[f"{info}/entry_points.txt"] = Path(args.entry_points).read_text()
            if args.license_file and Path(args.license_file).is_file():
                extras[f"{info}/LICENSE"] = Path(args.license_file).read_text(errors="replace")

            for dest, text in extras.items():
                data = text.encode()
                z.writestr(dest, data)
                digest = base64.urlsafe_b64encode(
                    hashlib.sha256(data).digest()).rstrip(b"=").decode()
                records.append((dest, f"sha256={digest}", len(data)))

            # RECORD lists itself with neither hash nor size
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            for row in sorted(records):
                w.writerow(row)
            w.writerow([f"{info}/RECORD", "", ""])
            z.writestr(f"{info}/RECORD", buf.getvalue())
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    total = sum(r[2] for r in records)
    print(f"  {whl}")
    print(f"  {len(chosen)} files, {total / 1048576:.1f} MB uncompressed, "
          f"{whl.stat().st_size / 1048576:.1f} MB compressed")
    if args.strip and raw:
        print(f"  stripped {raw / 1048576:.1f} -> {packed / 1048576:.1f} MB "
              f"({100 * (raw - packed) / raw:.0f}% off before compression)")
    if args.cuda_from_wheels:
        if cuda["patched"]:
            print(f"  CUDA from the pip wheels: RUNPATH added to "
                  f"{cuda['patched']} binaries")
        if cuda["skipped"]:
            print(f"  note: patchelf not found, so {cuda['skipped']} binaries "
                  f"keep whatever CUDA the loader finds -- `pip install "
                  f"patchelf` to point them at the nvidia wheels",
                  file=sys.stderr)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True, help="install prefix to take files from")
    p.add_argument("--contents", required=True, action="append",
                   help="a contents file; repeatable, so a generated selection "
                        "can be layered over the hand-written list")
    p.add_argument("--manifest",
                   help="CMake install_manifest.txt; restricts the wheel to what "
                        "this build installed, rather than whatever the prefix "
                        "has accumulated")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", default="viame")
    p.add_argument("--version", required=True)
    p.add_argument("--local-version",
                   help="a PEP 440 local segment, e.g. `cu12`, appended as "
                        "`+cu12`. The default build carries none. Note PyPI "
                        "refuses local versions, so a variant wheel needs an "
                        "index of its own.")
    p.add_argument("--summary", default="VIAME: Video and Image Analytics for Marine Environments")
    p.add_argument("--description", default="See https://github.com/VIAME/VIAME")
    p.add_argument("--requires", action="append", help="a Requires-Dist entry; repeatable")
    p.add_argument("--requires-python", default="3.10",
                   help="the floor for Requires-Python. Not below the wheel's "
                        "own python tag: it is built for one interpreter, and "
                        "vendored sam2 needs 3.10 regardless.")
    p.add_argument("--requires-from", action="append",
                   help="a file of Requires-Dist entries, one per line, `#` "
                        "comments ignored; kept as a file so the reasoning for "
                        "each can live beside it. Repeatable, so a CUDA "
                        "variant's list layers over the common one.")
    p.add_argument("--top-level", action="append", help="a top-level package name; repeatable")
    p.add_argument("--entry-points", help="an entry_points.txt to embed")
    p.add_argument("--license-file")
    p.add_argument("--python-tag", default=f"cp{sys.version_info.major}{sys.version_info.minor}")
    p.add_argument("--abi-tag", default=f"cp{sys.version_info.major}{sys.version_info.minor}")
    p.add_argument("--platform-tag", default="linux_x86_64")
    p.add_argument("--strip", action="store_true",
                   help="strip symbol tables from packed binaries; see `strip_into`")
    p.add_argument("--no-strip", dest="strip", action="store_false")
    p.set_defaults(strip=True)
    p.add_argument("--cuda-from-wheels", action="store_true",
                   help="add RUNPATH entries for the nvidia pip wheels, so "
                        "CUDA comes from the copy torch installed rather than "
                        "a system one; see `add_cuda_rpath`")
    p.add_argument("--no-cuda-from-wheels", dest="cuda_from_wheels",
                   action="store_false")
    p.add_argument("--cuda-major", type=int, choices=sorted(CUDA_WHEEL_DIRS),
                   help="the CUDA major this was built against, which decides "
                        "which nvidia wheel layout the RUNPATH targets. Omit "
                        "to target every known layout.")
    p.set_defaults(cuda_from_wheels=True)
    return build(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
