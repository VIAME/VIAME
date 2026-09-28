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

#: An import of OpenCV, as a diff line. `+` and `-` are stripped before this
#: is tried, so it matches the line in either direction.
IMPORT_LINE = re.compile(r"^\s*(?:import\s+cv2\b|from\s+cv2[\s.]|"
                         r"from\s+cv2$)")


def diff_verdicts(patches, fork):
    """What a fork's unified diff does about OpenCV, per file it touches.

    `True` for a file whose imports it **removes and does not add back**, and
    `False` for one it touches without doing that. The distinction is the
    point: exempting a file because *something* patches it would pass a diff
    that changed an unrelated line and left the import where it was, which is
    exactly the shape a careless rebase produces.
    """
    diff = os.path.join(patches, fork + ".patch")

    if not os.path.isfile(diff):
        return {}

    with open(diff, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    verdicts = {}
    path = None
    removed = added = 0

    def settle():
        if path is not None:
            verdicts[path] = removed > 0 and added == 0

    for line in lines:
        match = DIFF_TARGET.match(line)

        if match:
            settle()
            path = match.group(1)
            removed = added = 0
            continue

        if path is None or not line:
            continue

        if line[0] == "-" and not line.startswith("---"):
            if IMPORT_LINE.match(line[1:]):
                removed += 1
        elif line[0] == "+" and not line.startswith("+++"):
            if IMPORT_LINE.match(line[1:]):
                added += 1

    settle()

    return verdicts


def overlay_verdicts(patches, fork):
    """The same question for whole replacement files: does the copy import it?

    Easier than the diff case, because the overlay file **is** what ships, so
    it can simply be read.
    """
    overlay = os.path.join(patches, fork)
    verdicts = {}

    if not os.path.isdir(overlay):
        return verdicts

    for base, directories, names in os.walk(overlay):
        directories[:] = [d for d in directories if d != "__pycache__"]

        for name in names:
            path = os.path.join(base, name)
            relative = os.path.relpath(path, overlay)
            verdicts[relative] = not (name.endswith(".py")
                                      and imports_cv2(path))

    return verdicts


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

        covered = dict(diff_verdicts(patches, fork))
        covered.update(overlay_verdicts(patches, fork))

        for path in runtime_files(root):
            checked += 1
            offences = imports_cv2(path)

            if not offences:
                continue

            relative = os.path.relpath(path, root)
            verdict = covered.get(relative)

            if verdict is True:
                covered_total += 1
                continue

            for line, what in offences:
                if verdict is False:
                    failures.append(
                        "{}/{}:{}: `{}` survives the patch -- it touches this "
                        "file without taking the import out".format(
                            fork, relative, line, what))
                else:
                    failures.append(
                        "{}/{}:{}: `{}` and no patch replaces this file"
                        .format(fork, relative, line, what))

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
