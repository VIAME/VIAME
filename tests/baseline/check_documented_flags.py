#!/usr/bin/env python3
"""Every build flag README.md advertises is a flag the build declares.

A flag that is documented but not declared is worse than an undocumented
one: setting it is silent, so a user who turns a feature off gets it anyway
and has no way to tell. Two were found this way -- `VIAME_ENABLE_VXL`, whose
implementations became VIAME's own in phase 3, and `VIAME_ENABLE_TENSORRT`,
removed in P1 -- and both had outlived their `option()` by months.

The reverse is not checked. A declared flag that the README leaves out is a
documentation gap, not a trap, and the advanced ones are deliberately absent.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

# `VIAME_ENABLE_PYTORCH-*` and friends stand for a family whose members are
# declared individually; a row ending in `*` is checked as a prefix.
WILDCARD = "*"


def documented():
    """Flags named in the first column of README.md's flag tables."""
    found = {}
    for number, line in enumerate(
            (ROOT / "README.md").read_text(encoding="utf-8").splitlines(), 1):
        match = re.match(r"\|\s*(VIAME_[A-Za-z0-9_-]+\*?)\s*\|", line)
        if match:
            found.setdefault(match.group(1), number)
    return found


def declared():
    """Flags the build declares **or reads**.

    Reads count. A cache variable passed with `-D` works whether or not an
    `option()` names it, so `VIAME_BUILD_DIVE_FROM_SOURCE` -- read by
    `cmake/viame_dive.cmake` and declared nowhere -- is a working flag, not a
    dead one. Counting only declarations called it dead and nearly had it
    deleted.
    """
    names = set()
    for path in list((ROOT / "cmake").rglob("*.cmake")) + [ROOT / "CMakeLists.txt"]:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lstrip().startswith("#"):
                continue
            names.update(re.findall(r"option\(\s*(VIAME_[A-Za-z0-9_-]+)", line))
            names.update(re.findall(
                r"set\(\s*(VIAME_[A-Za-z0-9_-]+)[^)]*CACHE", line))
            names.update(re.findall(
                r"(?:if|elseif)\s*\(\s*(?:NOT\s+)?(VIAME_[A-Za-z0-9_-]+)", line))
            names.update(re.findall(r"\$\{(VIAME_[A-Za-z0-9_-]+)\}", line))
    return names


def main():
    real = declared()
    problems = []

    for flag, line in sorted(documented().items()):
        if flag.endswith(WILDCARD):
            prefix = flag[:-1]
            if not any(name.startswith(prefix) for name in real):
                problems.append(
                    f"README.md:{line}: `{flag}` matches no declared flag")
        elif flag not in real:
            problems.append(
                f"README.md:{line}: `{flag}` is documented but never declared")

    if problems:
        print("Documented flags the build does not declare:")
        for problem in problems:
            print("  " + problem)
        print("\nEither declare the flag or take the row out; a flag that only "
              "exists in the README is one a user can set and never notice "
              "doing nothing.")
        return 1

    print(f"every documented flag is declared ({len(documented())} checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
