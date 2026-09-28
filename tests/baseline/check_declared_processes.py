#!/usr/bin/env python3
"""Every python process the tree defines is one the tree declares.

On `main` a python process registers by being imported: the loader scans the
process packages and calls each module's `__sprokit_register__`. This branch
has no scan -- P8-T10 replaced it with declarations, so that `viame runner
--help` does not import torch -- and a process that is not declared in its
package's `__sprokit_process_declarations__` does not exist, however
complete its module is.

That makes every merge from `main` a way to lose one. It has happened:
`compute_curved_measurements` arrived with its class, its registration hook
and two add-on pipelines that select it, and no declaration. Nothing failed.
`baseline:pipes` had recorded the pipeline as already broken, so the
comparison agreed with itself.

Reads source, not an install, so it runs before anything is built.
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "library"


def declared():
    """Process names in any package's `__sprokit_process_declarations__`."""
    names = set()
    # Every python file, not only `__init__.py`: a package's init can be
    # installed from a file of another name, as `viame.processes` is from
    # `processes_init.py`.
    for path in LIBRARY.rglob("*.py"):
        if "tpl" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name)
                       and target.id == "__sprokit_process_declarations__"
                       for target in node.targets):
                continue
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                continue
            for entry in node.value.elts:
                if (isinstance(entry, ast.Tuple) and entry.elts
                        and isinstance(entry.elts[0], ast.Constant)):
                    names.add(entry.elts[0].value)
    return names


def defined():
    """(name, file) for every `add_process` inside a registration hook."""
    found = []
    for path in LIBRARY.rglob("*.py"):
        if "tpl" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hook = text.find("def __sprokit_register__")
        if hook < 0:
            continue
        for name in re.findall(
                r"add_process\(\s*['\"]([A-Za-z_0-9]+)['\"]", text[hook:]):
            found.append((name, path.relative_to(ROOT)))
    return found


def declaring_directories():
    """Directories whose package declares what it provides.

    `module_loader.load_python_modules` does not scan such a package -- "a
    package that declares what it provides is not scanned" -- so these are
    the only places a process can be lost. A package that declares nothing,
    `viame.processes` for one, is still scanned and its hooks still run.
    """
    found = set()
    for path in LIBRARY.rglob("*.py"):
        if "tpl" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^__(sprokit_process|vital_algorithm)_declarations__\s*=",
                     text, re.MULTILINE):
            found.add(path.parent)
    return found


def main():
    known = declared()
    unscanned = declaring_directories()
    missing = [(name, path) for name, path in defined()
               if name not in known and (ROOT / path).parent in unscanned]

    if missing:
        print("Python processes with a registration hook and no declaration:")
        for name, path in sorted(missing):
            print("  {:36s} {}".format(name, path))
        print("\nAdd each to `__sprokit_process_declarations__` in its "
              "package's __init__.py, as\n"
              "  ( name, description, \"module.path:ClassName\" ).\n"
              "Without it the process is not in the registry and every "
              "pipeline naming it fails to build.")
        return 1

    print("every python process is declared ({} checked)".format(len(defined())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
