#!/usr/bin/env python3
"""Vendor a python fork into the `viame` package, under a name we own.

Run at build time from `packages/pytorch-libs/<fork>`, so the submodule stays
a clean checkout and an upstream merge never conflicts with this rewrite.

Used for `rfdetr` and `sam2`. Both are forks whose PyPI name resolves to
different software, so the wheel can neither depend on the name nor ship its
own code under it.

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


def import_rules(name, target):
    """Rewrites for python import statements naming `name`.

    Ordered longest-first so `from x.y import` is not half-matched by the
    `from x import` rule.
    """
    n = re.escape(name)
    return [
        (re.compile(rf"(?m)^(\s*)from {n}\.(\S+) import "),
         lambda m: f"{m.group(1)}from {target}.{m.group(2)} import "),
        (re.compile(rf"(?m)^(\s*)from {n} import "),
         lambda m: f"{m.group(1)}from {target} import "),
        # `import a.b.c` without an `as` binds `a`; rewriting it would bind
        # `viame` instead and break every use of the old name. Checked for
        # each fork before adding it here.
        (re.compile(rf"(?m)^(\s*)import {n}\.(\S+) as "),
         lambda m: f"{m.group(1)}import {target}.{m.group(2)} as "),
        (re.compile(rf"(?m)^(\s*)import {n}\s*$"),
         lambda m: f"{m.group(1)}from {target.rsplit('.', 1)[0]} import {name}"),
    ]


def string_rule(name, target):
    """Module paths written as strings: `import_module("x.y")` and friends."""
    return (re.compile(rf"([\"']){re.escape(name)}\."), lambda m: f"{m.group(1)}{target}.")


def config_rules(name, target):
    """Dotted paths in configuration data, not python.

    sam2 resolves its model graph through Hydra, so 125 `_target_: sam2.x.Y`
    lines in its YAML name modules the same way an import does, and
    `initialize_config_module("sam2")` registers the config package itself by
    name. Renaming the package without these gives a package that imports and
    then fails at model construction, which is a worse failure than not
    building.
    """
    n = re.escape(name)
    return [
        (re.compile(rf"(_target_:\s*){n}\."), lambda m: f"{m.group(1)}{target}."),
        (re.compile(rf"(initialize_config_module\(\s*[\"']){n}([\"'])"),
         lambda m: f"{m.group(1)}{target}{m.group(2)}"),
    ]


def leftover_re(name):
    n = re.escape(name)
    return re.compile(rf"(?m)^\s*(from {n}[\s.]|import {n}[\s.]|import {n}$)")


def rewrite(text, name, target, is_config):
    """Returns (text, imports rewritten, other references rewritten)."""
    imports = other = 0
    if not is_config:
        for pattern, repl in import_rules(name, target):
            text, n = pattern.subn(repl, text)
            imports += n
        pattern, repl = string_rule(name, target)
        text, n = pattern.subn(repl, text)
        other += n
    for pattern, repl in config_rules(name, target):
        text, n = pattern.subn(repl, text)
        other += n
    return text, imports, other


CONFIG_SUFFIXES = (".yaml", ".yml")
TEXT_SUFFIXES = (".py",) + CONFIG_SUFFIXES


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="the fork's package directory")
    p.add_argument("--patches",
                   help="a directory of whole-file replacements to overlay on "
                        "the source before rewriting; `packages/patches/<fork>`")
    p.add_argument("--name", required=True, help="the fork's own package name, e.g. sam2")
    p.add_argument("--target", required=True, help="where it lands, e.g. viame.sam2")
    p.add_argument("--licence", help="LICENSE to copy beside the vendored code")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    source = Path(args.source)
    if not (source / "__init__.py").is_file():
        raise SystemExit(
            f"vendor_python_fork: {source} has no __init__.py.\n"
            f"  The submodule is probably not checked out:\n"
            f"    git submodule update --init <its path>")

    output = Path(args.output)
    staged = output.with_name(output.name + ".staging")
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir(parents=True)

    # The patch overlay first, so the rewrite sees patched text. A fork VIAME
    # patches has whole files copied over its source before it is built, and
    # vendoring the raw submodule instead would silently drop those edits.
    patched = set()
    if args.patches:
        patches = Path(args.patches)
        if not patches.is_dir():
            raise SystemExit(f"vendor_python_fork: {patches} is not a directory")
        for src in sorted(patches.rglob("*")):
            if src.is_dir() or "__pycache__" in src.parts:
                continue
            patched.add(src.relative_to(patches))

    leftover = leftover_re(args.name)
    files = imports = others = 0
    for src in sorted(source.rglob("*")):
        if "__pycache__" in src.parts:
            continue
        rel = src.relative_to(source)
        dst = staged / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel in patched:
            src = Path(args.patches) / rel        # the patched copy wins
        if src.suffix in TEXT_SUFFIXES:
            text = src.read_text(encoding="utf-8", errors="surrogateescape")
            text, n_imp, n_oth = rewrite(text, args.name, args.target,
                                         src.suffix in CONFIG_SUFFIXES)
            if src.suffix == ".py":
                bad = leftover.findall(text)
                if bad:
                    raise SystemExit(
                        f"vendor_python_fork: {rel} still imports `{args.name}` "
                        f"after the rewrite: {bad[:3]}\n"
                        f"  An import form this script does not know about has "
                        f"appeared upstream; add a rule for it.")
            dst.write_text(text, encoding="utf-8", errors="surrogateescape")
            files += 1
            imports += n_imp
            others += n_oth
        else:
            shutil.copy2(src, dst)

    if args.licence and Path(args.licence).is_file():
        shutil.copy2(args.licence, staged / "LICENSE")

    if output.is_dir() and _same(staged, output):
        shutil.rmtree(staged)
        print(f"  {args.target}: unchanged ({files} files)")
        return 0
    shutil.rmtree(output, ignore_errors=True)
    staged.rename(output)
    extra = f", {len(patched)} patched" if patched else ""
    print(f"  {args.target}: vendored {files} files{extra}, "
          f"{imports} imports and {others} module paths rewritten")
    return 0


def _same(a, b):
    a_files = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    b_files = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    if a_files != b_files:
        return False
    return all((a / r).read_bytes() == (b / r).read_bytes() for r in a_files)


if __name__ == "__main__":
    sys.exit(main())
