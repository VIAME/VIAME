#!/usr/bin/env python3
"""Move files from `plugins/` to `library/` per `lite-file-map.tsv`.

The map says where each file goes; this performs the move and repairs the
`#include` lines, which is the part that cannot be done by hand at this
scale. Three forms have to become one:

    #include "read_detected_object_set_dive.h"      same directory
    #include "../core/read_detected_object_set_dive.h"
    #include <plugins/core/read_detected_object_set_dive.h>

all become `#include <viame/file_io/read_detected_object_set_dive.h>`, except
where the including file lands in the same library as the header, which keeps
the quoted form.

Run it one capability at a time -- `--only file_io` -- so that each move is a
build and a test rather than one big bang.

Usage:
    apply_file_map.py --check              what is unmapped or already moved
    apply_file_map.py --only <dir> [--dry-run]
    apply_file_map.py --all [--dry-run]
"""

import argparse
import os
import re
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MAP = os.path.join(ROOT, "design", "lite-file-map.tsv")


def load_map():
    rows = []
    with open(MAP) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            source, destination = line.split("\t")
            rows.append((source, destination))
    return rows


def run(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("%s: %s" % (" ".join(args), result.stderr.strip()))
    return result.stdout


def ambiguous_basenames():
    """Header names that exist at more than one path.

    `camera_io.h` is `library/algorithm_framework/io/camera_io.h` as well as
    the one that came from `plugins/core`, and a neighbour's `#include
    "camera_io.h"` means the neighbour. Rewriting by basename cannot tell
    them apart, so it does not try.
    """
    seen = {}
    for path in run("git", "ls-files").split():
        if path.endswith((".h", ".hpp")):
            seen.setdefault(os.path.basename(path), []).append(path)
    return {name for name, paths in seen.items() if len(paths) > 1}


def header_destinations(rows):
    """{header basename: (library directory, new basename)} for headers that
    have already arrived.

    Only headers whose destination exists on disk: a capability is moved at a
    time, and pointing an include at a header that has not moved yet is a
    build break rather than a step forward.
    """
    found = {}
    ambiguous = ambiguous_basenames()
    for source, destination in rows:
        if destination in ("STRUCTURAL", "DELETE") or not source.endswith((".h", ".hpp")):
            continue
        if os.path.basename(destination) in ambiguous:
            continue
        if not os.path.exists(os.path.join(ROOT, destination)):
            continue
        # Still at its old path as well: a second, different file is mapped
        # onto a destination that already exists -- the opencv and core
        # `windowed_utils.h`, say, which P2-T05 merges. Rewriting an include
        # of that name now would pick whichever of the two the map happened
        # to mention, so leave every include of it alone until the merge.
        if os.path.exists(os.path.join(ROOT, source)):
            continue
        found[os.path.basename(source)] = (destination.split("/")[1],
                                           os.path.basename(destination))
    return found


def rewrite_includes(path, headers, owner):
    """Point every include of a moved header at its new home.

    `owner` maps a file path to the library directory it ends up in, so that
    a file which moves alongside its header keeps the quoted include.
    """
    full = os.path.join(ROOT, path)
    try:
        text = open(full, encoding="utf-8").read()
    except (UnicodeDecodeError, IsADirectoryError):
        return 0

    mine = owner.get(path)
    changed = 0

    def replace(match):
        nonlocal changed
        name = os.path.basename(match.group("name"))
        if name not in headers:
            return match.group(0)
        directory, base = headers[name]
        if mine == directory:
            new = '#include "%s"' % base
        else:
            new = "#include <viame/%s/%s>" % (directory, base)
        if new != match.group(0):
            changed += 1
        return new

    pattern = re.compile(
        r'#include\s*[<"](?P<name>[^">]*?[A-Za-z0-9_]+\.h(?:pp)?)[>"]')
    new_text = pattern.sub(replace, text)

    if changed:
        open(full, "w", encoding="utf-8").write(new_text)
    return changed


def package_of(path):
    """The importable package a python file belongs to, or None.

    `library/classifiers/x.py` is `viame.classifiers`;
    `plugins/pytorch/srnn/y.py` is `viame.pytorch.srnn`.
    """
    parts = path.split("/")
    if parts[0] == "library":
        return ".".join(["viame"] + parts[1:-1])
    if parts[0] == "plugins":
        return ".".join(["viame"] + parts[1:-1])
    return None


def fix_relative_imports():
    """Make a broken `from .x import` absolute.

    `multicam_homog_det_suppressor` is a classifier and `multicam_homog_tracker`
    is a tracker, and they were one package until the move split them. A
    relative import that no longer has a sibling to reach is the failure mode
    of every one of these moves, and it is an ImportError at plugin-discovery
    time rather than a build error, so it is worth repairing by rule.

    A relative import whose sibling is still beside the file is left alone:
    inside a vendored subtree -- netharn, loftr, remax -- every one of them is.
    """
    homes = {}
    for path in run("git", "ls-files", "library", "plugins").split():
        if path.endswith(".py"):
            homes.setdefault(os.path.basename(path)[:-3], []).append(path)

    pattern = re.compile(r"^(\s*)from \.([A-Za-z_][A-Za-z0-9_]*) import",
                         re.MULTILINE)
    fixed = 0

    for path in run("git", "ls-files", "library", "plugins").split():
        if not path.endswith(".py"):
            continue
        directory = os.path.dirname(path)
        package = package_of(path)
        if not package:
            continue

        def replace(match):
            nonlocal fixed
            name = match.group(2)
            beside = os.path.join(ROOT, directory, name)
            if os.path.exists(beside + ".py") or \
                    os.path.exists(os.path.join(beside, "__init__.py")):
                return match.group(0)
            # Only a sibling that moved into `library/`: that is what these
            # moves break. A name that happens to match something in an
            # unrelated vendored subtree is a coincidence -- and sometimes it
            # is not even code, but a `from .mixins import *` in a doctest.
            #
            # A module still at the top of a `plugins/<name>/` package counts
            # too: `stereo_algos` moved to `object_detectors` while the
            # `stereo_utils` it imports waits in `plugins/opencv` for P2-T06.
            candidates = [p for p in homes.get(name, [])
                          if (p.startswith("library/")
                              or os.path.dirname(p).count("/") == 1)
                          and package_of(p) != package]
            if len(candidates) != 1:
                return match.group(0)
            fixed += 1
            return "%sfrom %s.%s import" % (
                match.group(1), package_of(candidates[0]), name)

        original = open(os.path.join(ROOT, path), encoding="utf-8").read()
        updated = pattern.sub(replace, original)
        if updated != original:
            open(os.path.join(ROOT, path), "w", encoding="utf-8").write(updated)

    return fixed


def fix_package_imports():
    """Repoint `from viame.<old> import <module>` at the module's new package.

    The dotted form, `viame.core.detection_fusion_core`, is easy to rewrite
    with a search. The package form, `from viame.core import
    detection_fusion_core`, names the module after the `import` and slips
    past it -- and it fails only when the line runs, which for `tools/` and
    the fallback branches of a `try` can be a CRITICAL example rather than
    the build. Only a module name that exists at exactly one library root is
    rewritten.
    """
    homes = {}
    for path in run("git", "ls-files", "library").split():
        if (path.endswith(".py") and path.count("/") == 2
                and not path.endswith("__init__.py")
                and not path.startswith("library/tpl/")):
            homes.setdefault(os.path.basename(path)[:-3], set()).add(
                path.split("/")[1])

    pattern = re.compile(
        r"from viame\.(core|opencv|pytorch|onnx|colmap|svm|seagis) "
        r"import (\w+)")
    fixed = 0

    for path in run("git", "ls-files").split():
        if not path.endswith(".py") or path.startswith("library/tpl/"):
            continue
        full = os.path.join(ROOT, path)
        try:
            original = open(full, encoding="utf-8").read()
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            continue

        def replace(match):
            nonlocal fixed
            dirs = homes.get(match.group(2))
            if not dirs or len(dirs) != 1:
                return match.group(0)
            fixed += 1
            return "from viame.%s import %s" % (next(iter(dirs)),
                                                match.group(2))

        updated = pattern.sub(replace, original)
        if updated != original:
            open(full, "w", encoding="utf-8").write(updated)

    return fixed


def retarget_exports(path, directory):
    """Point a moved file's export macro and include guard at its new library.

    `viame_core_export.h` holding `VIAME_CORE_EXPORT` is generated for the
    target the file used to be built into. Built into `viame_utilities` it
    gets `viame_utilities_export.h` and `VIAME_UTILITIES_EXPORT`, and
    building against the old name is a missing header, so this is not
    optional tidying.
    """
    full = os.path.join(ROOT, path)
    try:
        text = open(full, encoding="utf-8").read()
    except UnicodeDecodeError:
        return False

    upper = directory.upper()
    original = text

    # The generated export header is found through the target's own include
    # directories, so a file that reached it by path -- it was generated into
    # the directory the file was in -- has to stop doing that.
    text = re.sub(r'<[a-z0-9_/]*/(viame_[a-z0-9_]+_export\.h)>', r'"\1"', text)

    # A process is built into `viame_processes_<dir>`, not `viame_<dir>`, and
    # its export header is named for the target it is in.
    text = re.sub(r'viame_processes_[a-z0-9_]+_export\.h',
                  "viame_processes_%s_export.h" % directory, text)
    text = re.sub(r'\bVIAME_PROCESSES_[A-Z0-9_]+_EXPORT\b',
                  "VIAME_PROCESSES_%s_EXPORT" % upper, text)

    text = re.sub(r'viame_(?!processes_)[a-z0-9_]+_export\.h',
                  "viame_%s_export.h" % directory, text)
    text = re.sub(r'\bVIAME_(?!PROCESSES_)[A-Z0-9_]+_EXPORT\b',
                  "VIAME_%s_EXPORT" % upper, text)

    # The include guard, taken from the file's own `#ifndef` rather than
    # guessed, so that a guard naming something other than the old library
    # (`VIAME_OPENCV_*` in a file that was in `plugins/core`) still moves.
    guard = re.search(r'^#ifndef +(VIAME_[A-Z0-9_]+_H(?:PP)?)$',
                      text, re.MULTILINE)
    if guard and not guard.group(1).startswith("VIAME_%s_" % upper):
        renamed = re.sub(r'^VIAME_[A-Z0-9]+_', "VIAME_%s_" % upper,
                         guard.group(1))
        text = text.replace(guard.group(1), renamed)

    if text != original:
        open(full, "w", encoding="utf-8").write(text)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="move only files destined for this library")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rewrite-only", action="store_true",
                        help="repair includes for files already moved")
    parser.add_argument("--from", dest="source_prefix",
                        help="only files under this path, e.g. plugins/core")
    parser.add_argument("--exclude", action="append", default=[],
                        help="skip sources matching this regex; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_map()
    tracked = set(run("git", "ls-files", "plugins").split())

    movable = [(s, d) for s, d in rows if d not in ("STRUCTURAL", "DELETE")]
    pending = [(s, d) for s, d in movable if s in tracked]
    done = len(movable) - len(pending)

    if args.check:
        unmapped = tracked - {s for s, _ in rows}
        print("map: %d movable, %d still in plugins/, %d already moved"
              % (len(movable), len(pending), done))
        print("unmapped files under plugins/: %d" % len(unmapped))
        for path in sorted(unmapped)[:20]:
            print("   " + path)
        return 1 if unmapped else 0

    if args.rewrite_only:
        selected = []
    elif args.only:
        selected = [(s, d) for s, d in pending
                    if d.split("/")[1] == args.only]
        if args.source_prefix:
            selected = [(s, d) for s, d in selected
                        if s.startswith(args.source_prefix)]
        for pattern in args.exclude:
            selected = [(s, d) for s, d in selected
                        if not re.search(pattern, s)]
        if not selected:
            print("nothing pending for library/%s" % args.only)
            return 0
    elif args.all:
        selected = pending
    else:
        parser.error("one of --check, --only, --all or --rewrite-only")

    print("moving %d files" % len(selected))

    # Which library each file is in *now*. A file still under `plugins/` has
    # no library, so its includes take the angle form -- using where it is
    # going to end up would make it include a neighbour it does not have yet.
    owner = {}

    if args.dry_run:
        for source, destination in selected[:20]:
            print("   %s -> %s" % (source, destination))
        if len(selected) > 20:
            print("   ... and %d more" % (len(selected) - 20))
        return 0

    for source, destination in selected:
        target = os.path.join(ROOT, os.path.dirname(destination))
        os.makedirs(target, exist_ok=True)
        run("git", "mv", source, destination)

    for _, destination in selected:
        if destination.endswith((".h", ".hpp", ".cxx", ".cpp", ".txx")):
            retarget_exports(destination, destination.split("/")[1])

    headers = header_destinations(rows)
    for path in run("git", "ls-files", "library").split():
        owner[path] = path.split("/")[1]

    touched = 0
    for path in run("git", "ls-files").split():
        if path.endswith((".h", ".hpp", ".cxx", ".cpp", ".txx", ".c")):
            touched += 1 if rewrite_includes(path, headers, owner) else 0

    relative = fix_relative_imports() + fix_package_imports()
    print("moved %d files; rewrote includes in %d; %d relative imports made "
          "absolute" % (len(selected), touched, relative))
    return 0


if __name__ == "__main__":
    sys.exit(main())
