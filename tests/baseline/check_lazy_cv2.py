#!/usr/bin/env python3
"""Reject direct OpenCV imports in VIAME's runtime Python code.

Calibration, display, and WLS now have local implementations, so deferring a
cv2 import no longer makes it acceptable. The historical test name is retained
for CI selectors. This source check does not cover transitive dependencies;
see design/cv2-removal-status.md for the remaining package work.

Tests (including reference recorders), templates, and external submodules are
excluded. Usage: check_lazy_cv2.py <source directory> [...]
"""
import argparse
import ast
import os
import sys


BANNED = "cv2"

# Files allowed a cv2 import, with the reason. Empty, and meant to
# stay that way: a new entry is a statement that a module cannot load without
# OpenCV, which is what the port exists to prevent.
ALLOWED = {}


def offending_nodes(tree):
    """Find direct OpenCV imports at any depth, including deferred imports."""
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


def dynamic_imports(tree):
    """`importlib.import_module("cv2")` and `__import__("cv2")` at any depth.

    These are reported wherever they appear, lazy or not, because a caller
    reaching for cv2 by name is worth seeing even when it is deferred -- and
    because no legitimate one exists in this tree.
    """
    found = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr

        if name not in ("import_module", "__import__"):
            continue

        for argument in node.args:
            if isinstance(argument, ast.Constant) and \
                    isinstance(argument.value, str) and (
                        argument.value == BANNED or
                        argument.value.startswith(BANNED + ".")):
                found.append((node.lineno, "%s(\"%s\")" % (name, BANNED)))

    return found


def python_files(root):
    """Every python file that is a module.

    `templates/` is skipped: its files carry `@template@` placeholders and are
    not valid python until a generator substitutes them, so a parse failure
    there is the file doing its job rather than a broken module.
    """
    for base, directories, names in os.walk(root):
        directories[:] = [d for d in directories
                          if d not in ("__pycache__", "packages", "tests",
                                       "templates")]
        for name in sorted(names):
            if name.endswith(".py"):
                yield os.path.join(base, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="directories to walk")
    arguments = parser.parse_args(argv)

    failures = []
    checked = 0

    for root in arguments.roots:
        for path in python_files(root):
            checked += 1
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError as error:
                failures.append("{}: cannot be parsed ({})".format(path, error))
                continue

            relative = os.path.relpath(path)
            if relative in ALLOWED:
                continue

            for line, what in offending_nodes(tree):
                failures.append(
                    "{}:{}: `{}` requires OpenCV; use a local implementation.".format(
                        relative, line, what))
            for line, what in dynamic_imports(tree):
                failures.append(
                    "{}:{}: `{}` reaches for OpenCV by name.".format(
                        relative, line, what))

    print("lazy cv2: checked {} python files".format(checked))

    if failures:
        for failure in failures:
            print("  " + failure)
        print("{} OpenCV import(s) or parse failure(s). See "
              "design/cv2-removal-status.md.".format(len(failures)))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
