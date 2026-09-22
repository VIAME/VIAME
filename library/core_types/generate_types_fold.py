#!/usr/bin/env python3
"""Generate the master translation unit for the folded `viame.types` module.

Run at configure time by `python.cmake`.

`viame.types` was fifty-six extension modules, which cost 68.6 MB because
every one carried its own copy of the same instantiated pybind11 and STL
templates; as a single module the code is 16.6 MB, 8.2 MB stripped. What
makes the fold more than a build change is **registration order**.

As separate modules the order did not matter. A binding that needed another's
types said so at runtime --

    py::module::import( "viame.types.camera" )

-- and python loaded it on demand. Folded, that line would re-enter the module
currently being initialised (`viame.types.camera` is now a shim that imports
`_types`), `PyInit__types` would run twice, and the second pass would fail on
the first duplicate registration with a message naming a type unrelated to the
cause:

    cannot initialize type "Image": an object with that name is already defined

So `VIAME_PYTHON_REQUIRE` is a no-op in a folded build and the requirement is
met by registering dependencies first instead. That order cannot be taken from
`types_init.py`: seventeen of the dependencies are back edges in its import
order. It is computed here, from the `VIAME_PYTHON_REQUIRE` lines themselves,
so that a binding that grows a new dependency is ordered correctly by saying
so in the one place it already had to.
"""

import argparse
import re
import sys
from pathlib import Path


MODULE_RE = re.compile(r'VIAME_PYTHON_MODULE\(\s*(\w+)\s*,', re.M)
REQUIRE_RE = re.compile(r'VIAME_PYTHON_REQUIRE\(\s*"viame\.types\.(\w+)"\s*\)')
INIT_RE = re.compile(r'^from viame\.types\.(\w+) import \*', re.M)


CLASS_RE = re.compile(r'py::class_<\s*(.*?)\s*>\s*(?:\(|\w)', re.S)


def _template_args(text):
    """Split a template argument list on commas at depth zero."""
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _leaf(type_name):
    """`kwiver::vital::track_state` -> `track_state`.

    The bindings spell the same type three ways -- `kv::track_state`,
    `viame::track_state`, and typedefs like `bbox` -- so the last identifier
    is what two declarations can be compared on.
    """
    type_name = re.sub(r"\bstd::shared_ptr<.*", "", type_name)
    return type_name.split("<")[0].strip().split("::")[-1].strip()


def base_class_edges(source_dir):
    """Dependencies implied by inheritance, which nothing declares.

    `py::class_< object_track_state, track_state >` cannot be registered
    before the module that registers `track_state`; pybind11 says

        type "ObjectTrackState" referenced unknown base type
        "viame::track_state"

    As separate modules this never arose -- the base's module was imported on
    demand -- so unlike the `VIAME_PYTHON_REQUIRE` lines there is no
    declaration of it anywhere, and it has to be read out of the
    `py::class_` template arguments.
    """
    owner, uses = {}, {}
    for path in sorted(Path(source_dir).glob("*_python.cxx")):
        text = path.read_text(errors="replace")
        m = MODULE_RE.search(text)
        if not m:
            continue
        module = m.group(1)
        for args in CLASS_RE.findall(text):
            parts = _template_args(args)
            if not parts:
                continue
            own = _leaf(parts[0])
            owner.setdefault(own, module)
            for base in parts[1:]:
                if "shared_ptr" in base or "unique_ptr" in base:
                    continue          # the holder, not a base
                leaf = _leaf(base)
                if leaf and leaf != own:
                    uses.setdefault(module, set()).add(leaf)

    edges = {}
    for module, bases in uses.items():
        for base in bases:
            home = owner.get(base)
            if home and home != module:
                edges.setdefault(module, set()).add(home)
    return edges


def collect(source_dir):
    """{module: {dependency, ...}}.

    Two sources, because there are two kinds of dependency and only one of
    them is declared: `VIAME_PYTHON_REQUIRE`, and inheritance.
    """
    modules = {}
    for path in sorted(Path(source_dir).glob("*_python.cxx")):
        text = path.read_text(errors="replace")
        m = MODULE_RE.search(text)
        if not m:
            continue                      # a helper translation unit
        modules[m.group(1)] = set(REQUIRE_RE.findall(text))

    for module, deps in base_class_edges(source_dir).items():
        if module in modules:
            modules[module] |= deps
    return modules


def order(modules, preferred):
    """A registration order: dependencies first, `preferred` breaking ties.

    Kahn's algorithm over the dependency edges, taking whichever ready module
    comes first in `preferred` -- the order `types_init.py` imports in -- so
    that the generated file stays stable and close to the order a reader of
    that file expects, and a rebuild does not reshuffle it.
    """
    rank = {name: i for i, name in enumerate(preferred)}
    pending = {m: set(d for d in deps if d in modules) for m, deps in modules.items()}
    done, out = set(), []
    while pending:
        ready = [m for m, deps in pending.items() if not (deps - done)]
        if not ready:
            cycle = " ".join(sorted(pending))
            raise SystemExit(
                f"generate_types_fold: dependency cycle among: {cycle}\n"
                f"  A cycle cannot be registered in one pass. Break it by "
                f"moving the shared type into a module both can depend on.")
        ready.sort(key=lambda m: (rank.get(m, len(rank)), m))
        pick = ready[0]
        out.append(pick)
        done.add(pick)
        del pending[pick]
    return out


TEMPLATE = '''// This file is part of VIAME, and is distributed under an OSI-approved #
// BSD 3-Clause License. See either the root top-level LICENSE file or  #
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
//
// GENERATED by generate_types_fold.py -- do not edit.
//
// The one extension module behind `viame.types`. Each translation unit that
// was its own module registers into a submodule of the same name, and
// `viame/types/<name>.py` re-exports it, so `viame.types.bounding_box` still
// names what it always did.
//
// The order below is a topological sort of the `VIAME_PYTHON_REQUIRE` lines
// in the bindings, not the order `types_init.py` imports in -- seventeen of
// these dependencies are back edges there. Every submodule registers during
// one import, so a dependency that has not been registered yet is a failure
// at import time rather than a missing module python can go and fetch.

#include <pybind11/pybind11.h>

%(decls)s

PYBIND11_MODULE( _types, m )
{
  m.doc() = "viame.types as one extension module. Reach the submodules "
            "through viame.types.<name>, which is what re-exports them.";

%(calls)s}
'''


def render(names):
    decls = "\n".join(f"void viame_register_python_{n}( ::pybind11::module& );"
                      for n in names)
    calls = "".join(
        f'  {{\n    auto sub = m.def_submodule( "{n}" );\n'
        f'    viame_register_python_{n}( sub );\n  }}\n' for n in names)
    return TEMPLATE % {"decls": decls, "calls": calls}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-modules", type=int, default=50,
                   help="fail below this many, so a regex that quietly stops "
                        "matching is caught here and not at import")
    args = p.parse_args(argv)

    modules = collect(args.source_dir)
    if len(modules) < args.min_modules:
        raise SystemExit(f"generate_types_fold: found {len(modules)} modules in "
                         f"{args.source_dir}, expected at least {args.min_modules}")

    init = Path(args.source_dir) / "types_init.py"
    preferred = INIT_RE.findall(init.read_text()) if init.is_file() else []

    missing = {d for deps in modules.values() for d in deps} - set(modules)
    if missing:
        print(f"  note: required but not modules here: {sorted(missing)}",
              file=sys.stderr)

    names = order(modules, preferred)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render(names)
    # Do not rewrite an identical file: the timestamp would rebuild the module
    # on every configure.
    if not out.is_file() or out.read_text() != text:
        out.write_text(text)
    print(f"  viame.types: {len(names)} modules folded into one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
