#!/usr/bin/env python3
"""Record the python API `kwiver.vital.types` presents.

P8-T01 replaces the bindings copied from kwiver with hand-written ones in
`library/core_types`, keeping the module name. That is a rewrite of a surface
that a great deal of python leans on -- every python algorithm, every python
sprokit process, the golden runner itself -- and none of it is type checked
until it runs.

So the surface is written down first. For every exported class: its name, its
methods and properties, and for each method whether it is a method or a
static one. Not the docstrings, which are not a contract, and not the C++
signatures, which pybind11 renders differently between versions.

    source <install>/setup_viame.sh
    python3 tests/golden/record_python_types.py

It refuses to overwrite without --force, for the reason every recorder here
does: a golden must not be quietly redefined by the code it checks.

What this cannot see is behaviour, and `tests/library/core_types/
test_python_types.py` beside it carries that -- the round trips and the
values that have to come back the same. The two together are the contract:
this one says what exists, that one says what it does.
"""
import argparse
import datetime
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "python_types.json")


def describe_class(cls):
    """The members of one bound class, by kind."""
    methods = []
    properties = []
    statics = []

    for name in dir(cls):
        if name.startswith("__") and name != "__init__":
            continue

        try:
            member = inspect.getattr_static(cls, name)
        except AttributeError:
            continue

        if isinstance(member, property):
            properties.append(name)
        elif isinstance(member, staticmethod):
            statics.append(name)
        elif callable(member) or inspect.isroutine(member):
            methods.append(name)
        else:
            # pybind11 renders a read-only attribute as a descriptor that is
            # neither; it is still part of the surface
            properties.append(name)

    return {
        "methods": sorted(set(methods)),
        "properties": sorted(set(properties)),
        "statics": sorted(set(statics)),
    }


def record():
    import kwiver.vital.types as types

    classes = {}
    values = {}

    for name in sorted(dir(types)):
        if name.startswith("_"):
            continue

        member = getattr(types, name)

        if inspect.isclass(member):
            classes[name] = describe_class(member)
        elif isinstance(member, (int, float, str, bool)):
            # module level constants, of which the metadata tags are many
            values[name] = member

    return {
        "recorded": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "What kwiver.vital.types exported before P8-T01 rewrote the "
                "bindings. Names and members only; behaviour is in "
                "tests/library/core_types/test_python_types.py.",
        "python": "{}.{}".format(*sys.version_info[:2]),
        "classes": classes,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing recording")
    args = parser.parse_args()

    if os.path.exists(TARGET) and not args.force:
        print("{} already recorded; pass --force to re-record".format(TARGET))
        return 1

    payload = record()

    with open(TARGET, "w") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")

    print("recorded {} classes and {} constants into {}".format(
        len(payload["classes"]), len(payload["values"]), TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
