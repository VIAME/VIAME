#!/usr/bin/env python3
"""No two VIAME libraries may define the same symbol.

The loader binds one definition and gives it to every caller in both
libraries. Nothing warns; the program links, loads and runs, and half the
callers silently get code they were not built against.

VIAME has been caught by this twice. `viame::enhance_images` was defined in
`plugins/vxl` and `plugins/opencv` with identical mangled names, and which
one both factories got depended on load order (`lite-findings.md` 1.10).
Then upstream added an OpenCV-free `plugins/core/windowed_utils` beside the
ported `plugins/opencv/windowed_utils`, and their `prepare_image_regions`
matched signature for signature: `ocv_windowed` quietly started running the
core implementation, which had an out-of-bounds write in its padding path,
and the black-pad golden case went from passing to a segfault.

Both were found by accident. This finds them on purpose.

Usage:
  check_symbols.py <library directory> [--pattern libviame_*.so*]
"""
import argparse
import collections
import fnmatch
import os
import subprocess
import sys


def defined_symbols(path):
    """The global text symbols this library defines, demangled."""
    try:
        out = subprocess.run(
            ["nm", "-DC", "--defined-only", path],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"could not read {path}: {error}")
        return set()

    found = set()

    for line in out.splitlines():
        # "<address> T <name>"; T is a global function in the text section,
        # which is what a duplicate definition collides on. Weak symbols (W,
        # V) are how templates and inlines are meant to be merged and are
        # not a conflict.
        parts = line.split(" T ")
        if len(parts) != 2:
            continue

        name = parts[1].strip()

        # VIAME's own code only. A library legitimately re-exports third
        # party symbols it static-links, and that is not this bug.
        if name.startswith("viame::") or name.startswith("kwiver::"):
            found.add(name)

    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="Directory holding the libraries")
    parser.add_argument("--pattern", default="libviame_*.so*",
        help="Which libraries to compare (default: libviame_*.so*)")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.directory):
        print(f"not a directory: {args.directory}")
        return 1

    # One real file per library: the versioned .so.1 and the .so beside it
    # are the same inode's worth of symbols under two names.
    libraries = {}

    for name in sorted(os.listdir(args.directory)):
        if not fnmatch.fnmatch(name, args.pattern):
            continue

        path = os.path.join(args.directory, name)
        if not os.path.isfile(path) or os.path.islink(path):
            continue

        libraries[name] = path

    if not libraries:
        print(f"no libraries matching {args.pattern} in {args.directory}")
        return 1

    owners = collections.defaultdict(list)

    for name, path in libraries.items():
        for symbol in defined_symbols(path):
            owners[symbol].append(name)

    clashes = {s: l for s, l in owners.items() if len(l) > 1}

    print(f"Checked {len(libraries)} libraries, "
          f"{len(owners)} distinct symbols")

    if not clashes:
        print("No symbol is defined by more than one library")
        return 0

    print(f"\n{len(clashes)} symbols are defined by more than one library:\n")

    for symbol, in_libraries in sorted(clashes.items()):
        print(f"  {symbol}")
        for name in sorted(in_libraries):
            print(f"      {name}")
        print()

    print("The loader binds one of each and every caller in both libraries "
          "gets it. Give the symbol one definition, or put the copies in "
          "namespaces that differ.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
