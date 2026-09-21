#!/usr/bin/env python3
"""Build a PEP 427 wheel from a VIAME install prefix.

The wheel's contents are *data*, not code: `contents.txt` beside this file
says which paths of the install prefix go in and where they land. That is the
whole reason this is written the way it is -- `main` installs a `kwiver`
package beside `viame` and 65 separate shared libraries, `lite` installs one
`viame` package and a single `libviame.so`, and the two branches have to share
this script so that merging one into the other is a change to a list rather
than a conflict in a builder.

Native libraries are not relocated and nothing is patched. VIAME's extension
modules already carry a `$ORIGIN`-relative RUNPATH --

    $ORIGIN/../../../../../lib

-- which, from `site-packages/<pkg>/<sub>/x.so`, resolves to the environment's
`lib/`. A wheel can put a file exactly there through the `.data/data/` scheme,
so the libraries go to `<name>-<version>.data/data/lib/` and the RUNPATH that
worked in the install prefix keeps working in the installed wheel. This is why
there is no `patchelf` dependency here; see `docs/wheels.md` for what that
costs, and for the layouts where it does not hold.
"""

import argparse
import base64
import csv
import hashlib
import io
import re
import sys
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


def select(prefix, rules, data_dir):
    """Return {wheel-relative path: source Path}.

    A destination may use `{data}`, which expands to the wheel's
    `<name>-<version>.data/data` -- the scheme pip unpacks into the
    environment prefix, and so the one place a wheel can put a shared library
    where an `$ORIGIN/../../../../../lib` RUNPATH will find it.
    """
    prefix = Path(prefix)
    if not prefix.is_dir():
        raise SystemExit(f"{prefix}: not a directory")

    chosen = {}
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
            rule_dest = rule.dest.replace("{data}", data_dir)
            if rule_dest.endswith("/"):
                # No `**` in the pattern means the glob names files in one
                # directory, so the file's own name is the tail.
                dest = rule_dest + (tail if tail is not None else src.name)
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

    if not chosen:
        raise SystemExit("the contents file selected no files")
    return chosen


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


def metadata(name, version, summary, requires, description):
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
        f"Summary: {summary}",
        "License: BSD-3-Clause",
        "Requires-Python: >=3.8",
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
    rules = read_contents(args.contents)
    dist = f"{args.name}-{args.version}"
    chosen = select(args.prefix, rules, f"{dist}.data/data")

    tag = f"{args.python_tag}-{args.abi_tag}-{args.platform_tag}"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    whl = out / f"{dist}-{tag}.whl"

    requires = [r for r in (args.requires or []) if r]
    records = []

    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
        for dest in sorted(chosen):
            src = chosen[dest]
            z.write(src, dest)
            digest, size = _hash(src)
            records.append((dest, digest, size))

        info = f"{dist}.dist-info"
        extras = {
            f"{info}/METADATA": metadata(args.name, args.version, args.summary,
                                         requires, args.description),
            f"{info}/WHEEL": wheel_metadata(False, tag),
            f"{info}/top_level.txt": "".join(t + "\n" for t in sorted(args.top_level or [])),
        }
        if args.entry_points and Path(args.entry_points).is_file():
            extras[f"{info}/entry_points.txt"] = Path(args.entry_points).read_text()
        if args.license_file and Path(args.license_file).is_file():
            extras[f"{info}/LICENSE"] = Path(args.license_file).read_text(errors="replace")

        for dest, text in extras.items():
            data = text.encode()
            z.writestr(dest, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((dest, f"sha256={digest}", len(data)))

        # RECORD lists itself with neither hash nor size
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        for row in sorted(records):
            w.writerow(row)
        w.writerow([f"{info}/RECORD", "", ""])
        z.writestr(f"{info}/RECORD", buf.getvalue())

    total = sum(r[2] for r in records)
    print(f"  {whl}")
    print(f"  {len(chosen)} files, {total / 1048576:.1f} MB uncompressed, "
          f"{whl.stat().st_size / 1048576:.1f} MB compressed")
    return 0



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", required=True, help="install prefix to take files from")
    p.add_argument("--contents", required=True, help="the contents file")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", default="viame")
    p.add_argument("--version", required=True)
    p.add_argument("--summary", default="VIAME: Video and Image Analytics for Marine Environments")
    p.add_argument("--description", default="See https://github.com/VIAME/VIAME")
    p.add_argument("--requires", action="append", help="a Requires-Dist entry; repeatable")
    p.add_argument("--top-level", action="append", help="a top-level package name; repeatable")
    p.add_argument("--entry-points", help="an entry_points.txt to embed")
    p.add_argument("--license-file")
    p.add_argument("--python-tag", default=f"cp{sys.version_info.major}{sys.version_info.minor}")
    p.add_argument("--abi-tag", default=f"cp{sys.version_info.major}{sys.version_info.minor}")
    p.add_argument("--platform-tag", default="linux_x86_64")
    return build(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
