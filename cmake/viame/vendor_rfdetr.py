#!/usr/bin/env python3
"""Vendor the RF-DETR fork into the `viame` package as `viame.rfdetr`.

Run at build time from `packages/pytorch-libs/rf-detr`, so the submodule stays
a clean checkout of `github.com/VIAME/rf-detr` and an upstream merge never
conflicts with this rewrite.

**Why vendor it at all.** `rfdetr` on PyPI is different software. VIAME builds
a fork whose `resolution` takes a `(height, width)` tuple where the public
1.10.1 takes an int, so a wheel that declared `rfdetr` as a requirement would
resolve to something that fails at

    1 validation error for RFDETRLargeConfig
    resolution: Input should be a valid integer [input_value=(960, 1728)]

Vendoring under `viame.rfdetr` means the wheel carries the fork it actually
needs and claims no name it does not own.

**Why rewrite the imports rather than alias.** The package has 267 absolute
self-imports and no relative ones -- every module says `from rfdetr.x import
y`. A `sys.modules["rfdetr"]` alias would make those work, and was the first
plan, but it is wrong in the case that matters: a user who also has the real
`rfdetr` installed would have the vendored code's internal imports resolve
against *their* copy through the ordinary path finder, silently mixing two
versions of a model implementation. Rewriting makes `viame.rfdetr` closed over
itself, and leaves a user's `import rfdetr` meaning theirs.

Apache 2.0, so redistribution is fine; `LICENSE` is copied beside the code.

What does *not* need rewriting, checked rather than assumed:

  * relative imports -- there are none.
  * `_RemovedModuleFinder` in `__init__.py`, which keys on `__name__` rather
    than a literal, so it follows the package to its new name by itself.
  * `get_version()`, which looks up the *distribution* `rfdetr` and returns
    `None` on `PackageNotFoundError`. Vendored there is no such distribution,
    and `None` is what it already does.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


TARGET = "viame.rfdetr"

# Import statements. Ordered longest-first so that `from rfdetr.x import` is
# not half-matched by the `from rfdetr import` rule.
IMPORT_RULES = [
    (re.compile(r"(?m)^(\s*)from rfdetr\.(\S+) import "),
     lambda m: f"{m.group(1)}from {TARGET}.{m.group(2)} import "),
    (re.compile(r"(?m)^(\s*)from rfdetr import "),
     lambda m: f"{m.group(1)}from {TARGET} import "),
    # Every one of these has an `as` clause, checked before writing this:
    # `import a.b.c` without one binds `a`, and rewriting it would bind
    # `viame` instead and break every use of the old name.
    (re.compile(r"(?m)^(\s*)import rfdetr\.(\S+) as "),
     lambda m: f"{m.group(1)}import {TARGET}.{m.group(2)} as "),
    (re.compile(r"(?m)^(\s*)import rfdetr\s*$"),
     lambda m: f"{m.group(1)}from viame import rfdetr"),
]

# Module paths written as strings: `import_module("rfdetr.variants")`,
# `exc.name.startswith("rfdetr.")`, and the deprecation shim's own names.
STRING_RULE = (re.compile(r"([\"'])rfdetr\."), lambda m: f"{m.group(1)}{TARGET}.")

# What must not survive: a reference to the package by its old name that would
# now resolve to a different distribution, or to nothing.
LEFTOVER = re.compile(r"(?m)^\s*(from rfdetr[\s.]|import rfdetr[\s.]|import rfdetr$)")


def rewrite(text):
    count = 0
    for pattern, repl in IMPORT_RULES:
        text, n = pattern.subn(repl, text)
        count += n
    pattern, repl = STRING_RULE
    text, n = pattern.subn(repl, text)
    return text, count, n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True,
                   help="the fork's src/rfdetr directory")
    p.add_argument("--licence", help="LICENSE to copy beside the vendored code")
    p.add_argument("--output", required=True,
                   help="where viame/rfdetr is assembled")
    args = p.parse_args(argv)

    source = Path(args.source)
    if not (source / "__init__.py").is_file():
        raise SystemExit(
            f"vendor_rfdetr: {source} has no __init__.py.\n"
            f"  The rf-detr submodule is probably not checked out:\n"
            f"    git submodule update --init packages/pytorch-libs/rf-detr")

    output = Path(args.output)
    staged = output.with_name(output.name + ".staging")
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)

    files = imports = strings = 0
    for src in sorted(source.rglob("*")):
        if "__pycache__" in src.parts:
            continue
        rel = src.relative_to(source)
        dst = staged / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".py":
            text = src.read_text(encoding="utf-8", errors="surrogateescape")
            text, n_imp, n_str = rewrite(text)
            leftover = LEFTOVER.findall(text)
            if leftover:
                raise SystemExit(
                    f"vendor_rfdetr: {rel} still imports `rfdetr` after the "
                    f"rewrite: {leftover[:3]}\n"
                    f"  An import form this script does not know about has "
                    f"appeared upstream; add a rule for it.")
            dst.write_text(text, encoding="utf-8", errors="surrogateescape")
            files += 1
            imports += n_imp
            strings += n_str
        else:
            shutil.copy2(src, dst)

    if args.licence and Path(args.licence).is_file():
        shutil.copy2(args.licence, staged / "LICENSE")

    # Swap in only if something changed, so a configure does not rebuild the
    # whole vendored package on every run.
    if output.is_dir() and _same(staged, output):
        shutil.rmtree(staged)
        print(f"  viame.rfdetr: unchanged ({files} modules)")
        return 0
    shutil.rmtree(output, ignore_errors=True)
    staged.rename(output)
    print(f"  viame.rfdetr: vendored {files} modules, "
          f"{imports} imports and {strings} module paths rewritten")
    return 0


def _same(a, b):
    a_files = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    b_files = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    if a_files != b_files:
        return False
    return all((a / r).read_bytes() == (b / r).read_bytes() for r in a_files)


if __name__ == "__main__":
    sys.exit(main())
