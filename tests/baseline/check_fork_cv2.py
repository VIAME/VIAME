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
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
import sys


BANNED = "cv2"

#: Not part of an installed wheel's importable modules. `checks/` and
#: `demo/` are scripts a developer runs by hand, `.mim` is mim's copy of the
#: configs and demos, and a `setup.py` runs at build time where cv2 being
#: absent is already handled.
SKIP_DIRECTORIES = ("__pycache__", "tests", "test", "checks", "demo", "demos",
                    "docs", ".mim", ".git", "build", "dist")

@contextmanager
def patched_tree(root, patches, fork):
    """Yield the installed source view, without modifying the working tree.

    Match the build order: whole-file overlays, then the unified diff. The
    source may already contain that diff from a previous build; reverse-check
    it before applying. Always inspect the resulting files, including additions.
    """
    overlay = os.path.join(patches, fork)
    diff = os.path.abspath(os.path.join(patches, fork + ".patch"))
    if not os.path.isdir(overlay) and not os.path.isfile(diff):
        yield root
        return

    with tempfile.TemporaryDirectory(prefix="viame-fork-cv2-") as temporary:
        staged = os.path.join(temporary, "source")
        # Dereference symlinks while copying so an overlay cannot write through
        # a symlink into the original checkout. Never copy git metadata.
        excluded = shutil.ignore_patterns(".git", "__pycache__", "build", "dist")

        def ignored(directory, names):
            omitted = excluded(directory, names)
            # Some upstream documentation has dangling image links. They are
            # not Python inputs; missing Python links still fail preparation.
            for name in names:
                path = os.path.join(directory, name)
                if (not name.endswith(".py") and os.path.islink(path)
                        and not os.path.exists(path)):
                    omitted.add(name)
            return omitted

        shutil.copytree(root, staged, ignore=ignored)
        if os.path.isdir(overlay):
            shutil.copytree(overlay, staged, dirs_exist_ok=True, ignore=ignored)
        if os.path.isfile(diff):
            env = {key: value for key, value in os.environ.items()
                   if key not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE",
                                  "GIT_COMMON_DIR")}

            def git(*args):
                return subprocess.run(["git", *args], cwd=staged, env=env,
                                      capture_output=True, text=True)

            # Isolate discovery even when TMPDIR is inside a git checkout.
            initialized = git("init", "--quiet")
            if initialized.returncode:
                raise RuntimeError(initialized.stderr.strip())
            reverse = git("apply", "--reverse", "--check",
                          "--ignore-whitespace", diff)
            if reverse.returncode:
                applied = git("apply", "--ignore-whitespace", diff)
                if applied.returncode:
                    raise RuntimeError("cannot apply {}: {}".format(
                        diff, applied.stderr.strip()))
        yield staged


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

    for fork in sorted(os.listdir(forks)):
        root = os.path.join(forks, fork)

        if not os.path.isdir(root):
            continue

        if not os.listdir(root):
            skipped.append(fork)
            continue

        try:
            with patched_tree(root, patches, fork) as installed:
                for path in runtime_files(installed):
                    checked += 1
                    relative = os.path.relpath(path, installed)
                    for line, what in imports_cv2(path):
                        failures.append(
                            "{}/{}:{}: `{}` remains after patching".format(
                                fork, relative, line, what))
        except (OSError, RuntimeError) as error:
            failures.append("{}: {}".format(fork, error))

    print("fork cv2: checked {} runtime python files in {} forks".format(
        checked, len(os.listdir(forks)) - len(skipped)))

    if skipped:
        print("  skipped, never checked out: {}".format(", ".join(skipped)))

    if failures:
        for failure in failures:
            print("  " + failure)
        print("{} vendored OpenCV import(s) or patch preparation failure(s). "
              "Patch the file into packages/patches, or say in "
              "design/STATUS.md why the declaration has to stay.".format(
                  len(failures)))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
