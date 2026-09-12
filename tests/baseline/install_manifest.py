#!/usr/bin/env python3
"""What VIAME installs, and where.

The CMake helpers decide this: which directory a library lands in, which
headers are public, what goes in `lib/cmake`, where a python module is
written. P8-T08 replaces those helpers, and nothing in the tree checks their
output -- a build that installs a header one directory to the left compiles,
links, tests green, and breaks the next person who includes it.

So the surface is written down. Only VIAME's own files: `lib/*.so*`,
`include/`, `bin/`, `lib/cmake/`, and the `viame` and `kwiver` python
packages. Not `lib/python3.10/site-packages` at large -- 75,000 files of
torch and its dependencies, which VIAME does not place and cannot regress.

Usage:
  install_manifest.py <install prefix> --record manifest.txt
  install_manifest.py <install prefix> --check  manifest.txt
"""
import argparse
import os
import sys

# Where VIAME's own helpers put things. A path outside these is either a
# dependency's or not installed by us.
ROOTS = (
    "include",
)

# `bin/` is mostly pip console scripts -- `accelerate`, `torchrun` and a
# hundred others -- which VIAME does not place and which move whenever a
# python dependency is bumped. Only the executables VIAME's own build
# produces are recorded.
BINARIES = (
    "viame",
    "kwiver",
    "dump_klv.py",
    "demo_macro_magic",
    "demo_python_impl_call",
)

# Likewise `lib/cmake`: `opencv4` and `proj4` belong to fletch.
CMAKE_PACKAGES = (
    "viame",
    "kwiver",
    "sprokit",
)

SITE_PACKAGES = "lib/python3.10/site-packages"
PACKAGES = ("viame", "kwiver")


def installed(prefix):
    """Every file VIAME's own build places, as prefix-relative paths."""
    found = set()

    # Shared libraries sit directly in lib/; everything else under it belongs
    # to a dependency (postgresql, libpng, the python tree).
    lib = os.path.join(prefix, "lib")
    if os.path.isdir(lib):
        for name in os.listdir(lib):
            if ".so" in name and os.path.isfile(os.path.join(lib, name)):
                found.add("lib/" + name)

    for root in ROOTS:
        base = os.path.join(prefix, root)
        for directory, _, names in os.walk(base):
            for name in names:
                path = os.path.join(directory, name)
                found.add(os.path.relpath(path, prefix))

    for name in BINARIES:
        if os.path.isfile(os.path.join(prefix, "bin", name)):
            found.add("bin/" + name)

    for package in CMAKE_PACKAGES:
        base = os.path.join(prefix, "lib", "cmake", package)
        for directory, _, names in os.walk(base):
            for name in names:
                path = os.path.join(directory, name)
                found.add(os.path.relpath(path, prefix))

    for package in PACKAGES:
        base = os.path.join(prefix, SITE_PACKAGES, package)
        for directory, _, names in os.walk(base):
            for name in names:
                # Bytecode is a side effect of running, not of installing.
                if name.endswith(".pyc"):
                    continue
                path = os.path.join(directory, name)
                found.add(os.path.relpath(path, prefix))

    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="install prefix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--record", help="write the manifest here")
    group.add_argument("--check", help="compare against this manifest")
    args = parser.parse_args()

    found = installed(args.prefix)

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
