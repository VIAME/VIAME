#!/usr/bin/env python3
"""What VIAME's own build installs, and where.

The CMake helpers decide this: which directory a library lands in, which
headers are public, what goes in `lib/cmake`, where a python module is
written. P8-T08 replaced those helpers, and nothing in the tree checked
their output -- a build that installs a header one directory to the left
compiles, links, tests green, and breaks the next person who includes it.

**The set comes from the install log, not from the tree.** VIAME installs
into a prefix it shares with fletch, which puts thousands of headers of its
own there -- `include/cppdb`, GDAL's, OpenCV's -- and `make install` only
ever adds, so the tree also holds whatever older builds left behind. Reading
the directory therefore answers "what has accumulated here", which is not the
question. `cmake --install` names every file it places, one per line, and
that is the answer.

Usage:
  make install > install.log
  install_manifest.py install.log --prefix <prefix> --record manifest.txt
  install_manifest.py install.log --prefix <prefix> --check  manifest.txt
"""
import argparse
import os
import re
import sys

LINE = re.compile(r"^-- (?:Installing|Up-to-date): (.+)$", re.M)


def installed(log_path, prefix):
    """Every file this install placed, as prefix-relative paths."""
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        log = handle.read()

    prefix = os.path.abspath(prefix)
    found = set()

    for match in LINE.finditer(log):
        path = match.group(1).strip()
        if not path.startswith(prefix):
            # A superbuild stage installing somewhere else entirely.
            continue
        relative = os.path.relpath(path, prefix)
        # Bytecode is a side effect of running, not of installing.
        if relative.endswith(".pyc"):
            continue
        found.add(relative)

    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="output of the install step")
    parser.add_argument("--prefix", required=True, help="install prefix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", help="write the manifest here")
    group.add_argument("--check", help="compare against this manifest")
    args = parser.parse_args()

    found = installed(args.log, args.prefix)

    if not found:
        print("no install lines in {}: did the install run?".format(args.log))
        return 1

    if args.record:
        with open(args.record, "w") as handle:
            for path in sorted(found):
                handle.write(path + "\n")
        print("recorded {} paths".format(len(found)))
        return 0

    with open(args.check) as handle:
        expected = {line.strip() for line in handle if line.strip()}

    missing = sorted(expected - found)
    added = sorted(found - expected)

    for path in missing:
        print("  gone:  " + path)
    for path in added:
        print("  new:   " + path)

    if missing or added:
        print("install manifest: {} gone, {} new".format(
            len(missing), len(added)))
        print("If a change is intended, re-record and say why in "
              "design/STATUS.md.")
        return 1

    print("install manifest: {} paths, unchanged".format(len(expected)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
