#!/usr/bin/env python3
"""No vendored package reaches OpenCV in the python a build installs.

`check_lazy_cv2.py` beside this covers VIAME's own tree and skips
`packages/`, because those are submodules nobody here edits. They are still
the reason `opencv-python-headless` was declared: mmcv, mmdetection, imgaug,
mmdeploy and sam2 imported cv2 from twenty-eight runtime modules between them,
and `import mmdet` failed without it.

That is fixed by the patches in `packages/patches` -- a `<fork>.patch` unified
diff, a `<fork>/` directory of whole replacement files, or both -- which
`cmake/viame_python_forks.cmake` applies before the wheel is built. **So the
question this asks is not "does the source import cv2" but "does anything a
build installs import cv2"**, and a file the patch or the overlay covers is
answered by what replaces it.

Which matters because a submodule bump is where this rots. Upstream adds a cv2
call to a file nothing patches, the patch still applies, the wheel still
builds, and the declaration can never come out again. This says so at the bump
rather than a release later.

A fork whose submodule has never been checked out is skipped: an empty
directory is the normal state of a checkout that does not need it.

Usage: check_fork_cv2.py <packages directory>
"""
import argparse
import ast
import os
import re
import sys


BANNED = "cv2"

#: Not part of an installed wheel's importable modules. `checks/` and
#: `demo/` are scripts a developer runs by hand, `.mim` is mim's copy of the
#: configs and demos, and a `setup.py` runs at build time where cv2 being
#: absent is already handled.
SKIP_DIRECTORIES = ("__pycache__", "tests", "test", "checks", "demo", "demos",
                    "docs", ".mim", ".git", "build", "dist")

DIFF_TARGET = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.M)


def patched_paths(patches, fork):
    """Every path a fork's patch or overlay replaces, fork-relative."""
    covered = set()

    diff = os.path.join(patches, fork + ".patch")

    if os.path.isfile(diff):
        with open(diff, encoding="utf-8", errors="replace") as handle:
            for match in DIFF_TARGET.finditer(handle.read()):
                covered.add(match.group(1))

    overlay = os.path.join(patches, fork)

    if os.path.isdir(overlay):
        for base, directories, names in os.walk(overlay):
            directories[:] = [d for d in directories if d != "__pycache__"]
            for name in names:
                covered.add(os.path.relpath(os.path.join(base, name), overlay))

    return covered


def imports_cv2(path):
    """The lines of `path` that reach OpenCV, by import or by name."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        source = handle.read()

    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        # Upstream python this build does not have to parse -- a python 2 file
        # in a vendored tree, say. Not this test's business.
        return []

    found = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == BANNED or alias.name.startswith(BANNED + "."):
                    found.append((node.lineno, "import " + alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == BANNED or
                                node.module.startswith(BANNED + ".")):
                found.append((node.lineno, "from %s import" % node.module))

    return found


def runtime_files(root):
    """Every python file in a fork that an installed wheel would import."""
    for base, directories, names in os.walk(root):
        directories[:] = [d for d in directories
                          if d not in SKIP_DIRECTORIES
                          and not d.startswith("test")]
        for name in sorted(names):
            if not name.endswith(".py"):
                continue
            if name == "setup.py" or name.startswith("test_"):
                continue
            yield os.path.join(base, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", help="the packages directory")
    arguments = parser.parse_args(argv)

    patches = os.path.join(arguments.packages, "patches")
    forks = os.path.join(arguments.packages, "pytorch-libs")

    if not os.path.isdir(forks):
        print("fork cv2: no {}, nothing to check".format(forks))
        return 0

    failures = []
    checked = 0
    skipped = []
    covered_total = 0

    for fork in sorted(os.listdir(forks)):
        root = os.path.join(forks, fork)

        if not os.path.isdir(root):
            continue

        if not os.listdir(root):
            skipped.append(fork)
            continue

        covered = patched_paths(patches, fork)

        for path in runtime_files(root):
            checked += 1
            offences = imports_cv2(path)

            if not offences:
                continue

            relative = os.path.relpath(path, root)

            if relative in covered:
                covered_total += 1
                continue

            for line, what in offences:
                failures.append(
                    "{}/{}:{}: `{}` and no patch replaces this file".format(
                        fork, relative, line, what))

    print("fork cv2: checked {} runtime python files in {} forks".format(
        checked, len(os.listdir(forks)) - len(skipped)))

    if covered_total:
        print("  {} file(s) import cv2 in the submodule and are replaced by a "
              "patch".format(covered_total))

    if skipped:
        print("  skipped, never checked out: {}".format(", ".join(skipped)))

    if failures:
        for failure in failures:
            print("  " + failure)
        print("{} vendored OpenCV import(s) with nothing to replace them. "
              "Patch the file into packages/patches, or say in "
              "design/STATUS.md why the declaration has to stay.".format(
                  len(failures)))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
