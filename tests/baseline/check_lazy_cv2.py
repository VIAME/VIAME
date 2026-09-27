#!/usr/bin/env python3
"""No VIAME python module may import cv2 at module scope.

The whole point of the cv2 removal is that importing VIAME does not pull
OpenCV in. Every call site that still needs it -- highgui, the WLS disparity
filter, two pose solvers -- keeps its `import cv2` *inside* the function that
needs it, so the module loads and only the branch that cannot be ported pays.
That invariant is easy to break by habit and invisible when it is: the import
succeeds on a machine that has cv2, and the suite is green.

So this checks the source rather than the behaviour. A module-scope import is
one that runs when the module is loaded -- at the top of the file, or inside a
`try`, an `if`, or a class body -- as against one inside a function or method,
which runs when that function is called.

Caught by this and not by anything else: a lazy import moved back to the top
during an unrelated edit, and a `from cv2 import ...` added where a grep for
`import cv2` would not have found it.

`tests/` is exempt on purpose. The golden recorders call cv2 to produce the
reference a port is held to, and a review probe may import it directly;
neither ships. `packages/` is exempt because it is submodules.

Usage:
  check_lazy_cv2.py <source directory> [<source directory> ...]
"""
import argparse
import ast
import os
import sys


BANNED = "cv2"

# Files allowed a module-scope import, with the reason. Empty, and meant to
# stay that way: a new entry is a statement that a module cannot load without
# OpenCV, which is what the port exists to prevent.
ALLOWED = {}


def offending_nodes(tree):
    """Every cv2 import that runs when the module is loaded.

    Walks the module body rather than the whole tree, descending only through
    the statements that execute at import time. A function or class *body* is
    not descended into -- a method's import is lazy -- but a class body is,
    since it runs immediately.
    """
    found = []

    def visit(body):
        for node in body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == BANNED or \
                            alias.name.startswith(BANNED + "."):
                        found.append((node.lineno, "import " + alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module == BANNED or
                                    node.module.startswith(BANNED + ".")):
                    found.append((node.lineno, "from %s import" % node.module))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Lazy: runs when called.
                continue
            elif isinstance(node, ast.ClassDef):
                visit(node.body)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For,
                                   ast.While)):
                visit(node.body)
                visit(getattr(node, "orelse", []))
                visit(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    visit(handler.body)
                for item in getattr(node, "items", []):
                    del item

    visit(tree.body)
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
                    argument.value == BANNED:
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
                    "{}:{}: `{}` runs at import. Move it inside the function "
                    "that needs it.".format(relative, line, what))
            for line, what in dynamic_imports(tree):
                failures.append(
                    "{}:{}: `{}` reaches for OpenCV by name.".format(
                        relative, line, what))

    print("lazy cv2: checked {} python files".format(checked))

    if failures:
        for failure in failures:
            print("  " + failure)
        print("{} module-scope cv2 import(s). Every remaining call site keeps "
              "its import inside the function that needs it -- see "
              "design/lite-findings.md.".format(len(failures)))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
