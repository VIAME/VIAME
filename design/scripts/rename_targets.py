#!/usr/bin/env python3
"""Rename the kwiver-era CMake targets to `viame_*` (P11-T01, second pass).

    vital                  -> viame_algorithm_framework
    vital_types            -> viame_core_types
    vital_config           -> viame_config
    vital_logger           -> viame_logger
    vital_exceptions       -> viame_exceptions
    vital_util             -> viame_util
    vital_vpm              -> viame_plugin
    vital_algo             -> viame_algo
    vital_applets          -> viame_applets
    sprokit_pipeline       -> viame_pipeline_framework
    sprokit_pipeline_util  -> viame_pipeline_util
    kwiver_adapter         -> viame_adapter
    kwiver_processes_adapter -> viame_processes_adapter
    kwiver_epx_test        -> viame_epx_test
    sprokit_applets        -> viame_applets_plugin
    vital_python_util      -> viame_python_util
    sprokit_python_util    -> viame_pipeline_python_util

Each name follows the library directory the target is built in, and none
collides with the `viame_*` targets that already exist.

A target name is not only a target name. `viame_add_library` calls
`generate_export_header`, which derives the macro and the header from it, so
renaming `vital_types` renames `VITAL_TYPES_EXPORT` (174 uses),
`VITAL_TYPES_NO_EXPORT`, `VITAL_TYPES_DEPRECATED` and the file
`vital_types_export.h` (53 includes) with it. This rewrites all three, plus
the `viame_lite_generated( <file> <subdir> )` calls that name the files.

What it does not touch:

- `kwiver::<target>`, which is the shim decision 8 keeps for one release. The
  `add_library( kwiver::x ALIAS x )` lines and the facade list keep emitting
  the old names; only what they point at is renamed.
- `vital_config.h`, the configured compiler-options header, is renamed
  separately to `viame_compiler_config.h`: it is not an export header, and
  `viame_config.h` would read like the `viame_config` target's own.

Usage:
    rename_targets.py [--check] [PATH ...]

`--check` reports what would change and writes nothing.
"""

import argparse
import collections
import os
import re
import subprocess
import sys

# target -> new target
TARGETS = {
    "vital": "viame_algorithm_framework",
    "vital_types": "viame_core_types",
    "vital_config": "viame_config",
    "vital_logger": "viame_logger",
    "vital_exceptions": "viame_exceptions",
    "vital_util": "viame_util",
    "vital_vpm": "viame_plugin",
    "vital_algo": "viame_algo",
    "vital_applets": "viame_applets",
    "sprokit_pipeline": "viame_pipeline_framework",
    "sprokit_pipeline_util": "viame_pipeline_util",
    "kwiver_adapter": "viame_adapter",
    "kwiver_processes_adapter": "viame_processes_adapter",
    "kwiver_epx_test": "viame_epx_test",
    "sprokit_applets": "viame_applets_plugin",
    "vital_python_util": "viame_python_util",
    "sprokit_python_util": "viame_pipeline_python_util",
}

# Files whose names are not generated from a target, and so are renamed
# outright rather than as a side effect. `vital_config.h` is the configured
# compiler-options header -- `viame_config.h` would read like the
# `viame_config` target's own -- and `vital_types.h` is a hand-written header
# that happens to share the `vital_types` target's name.
FILES = {
    "vital_config.h": "viame_compiler_config.h",
    "vital_config.h.in": "viame_compiler_config.h.in",
    "vital_types.h": "viame_core_types.h",
}

# `SPROKIT_NO_EXPORT` and `VITAL_EXPORT` belong to `sprokit_pipeline` and
# `vital`: CMake's generate_export_header shortens a target whose name the
# macro prefix already carries. Spelled out so nothing is guessed.
EXTRA_MACROS = {
    "VITAL_EXPORT": "VIAME_ALGORITHM_FRAMEWORK_EXPORT",
    "VITAL_NO_EXPORT": "VIAME_ALGORITHM_FRAMEWORK_NO_EXPORT",
    "SPROKIT_NO_EXPORT": "VIAME_PIPELINE_FRAMEWORK_NO_EXPORT",
}

# No `.py` and no `.rst`: a python file's `vital` is the package directory
# `python/kwiver/vital`, which follows the directory rather than the target,
# and renaming `from . import vital` breaks every import in the install.
SUFFIXES = (".h", ".cxx", ".hpp", ".txx", ".cpp", ".in", ".cmake", ".txt")

# `design/` and `library/compat/` write `vital` as a namespace name, not a
# target: the namespace scripts carry it in their rules and fixtures, and
# the compat header aliases `kwiver::vital` to `viame`. Renaming the word
# there breaks the shim quietly.
SKIP_PREFIXES = ("packages/", "library/tpl/", "design/", "library/compat/")

# Two places hold the old names on purpose, and a second run would take them
# away: the alias loop that maps `vital_algo` to `viame_algo` behind
# `kwiver::vital_algo`, and the list the installed config package defines those
# names from. Both are the shim decision 8 keeps for one release.
SKIP_REGIONS = (
    ("library/algorithm_framework/CMakeLists.txt", "foreach( _viame_lite_pair", ")"),
    ("cmake/viame_project.cmake", "set( viame_shim_names", ")"),
)


def rules():
    """Every literal rewrite, longest first so no prefix eats another."""
    out = []
    for old, new in TARGETS.items():
        # The generated header and its macros, before the bare target name.
        out.append((old + "_export.h", new + "_export.h"))
        for suffix in ("_EXPORT", "_NO_EXPORT", "_DEPRECATED", "_BUILD_AS_STATIC",
                       "_DEPRECATED_EXPORT", "_DEPRECATED_NO_EXPORT"):
            out.append((old.upper() + suffix, new.upper() + suffix))
        out.append((old, new))
    out += list(FILES.items())
    out += list(EXTRA_MACROS.items())
    out.sort(key=lambda pair: len(pair[0]), reverse=True)
    return out


RULES = rules()


def tracked(paths):
    if paths:
        return sorted(p for p in paths if p.endswith(SUFFIXES))
    listed = subprocess.run(["git", "ls-files"], capture_output=True,
                            text=True, check=True).stdout.split()
    return sorted(f for f in listed
                  if f.endswith(SUFFIXES) and not f.startswith(SKIP_PREFIXES))


# `add_subdirectory( vital )` names the directory `python/kwiver/vital`, not
# the target. It is the only directory in the tree that shares an old target
# name, and renaming the argument stops CMake finding it.
SUBDIR_CALL = re.compile(r"(add_subdirectory\s*\(\s*)([A-Za-z_0-9]+)(\s*\))")

# `viame_add_python_library` builds its target as `python-<module path>-<name>`
# with the slashes turned into dots, so the directory `vital/modules` makes
# `python-vital.modules-modules`. The `vital` in it is the directory, and a
# reference has to match what the generator emits.
DERIVED_TARGET = re.compile(r"python-[A-Za-z_0-9.]+")

# `viame_add_python_module( <file> vital vital_logging )`: the second argument
# is where the module installs under `site-packages/kwiver`, which follows the
# directory, not the library target.
PYTHON_MODULE_CALL = re.compile(
    r"(viame_add_python_(?:module|library)\s*\(\s*\S+\s*\n?\s*)([A-Za-z_0-9/]+)", re.M)


def protected(text, path):
    """The spans in `path` that hold old names on purpose."""
    spans = []
    for where, start, end in SKIP_REGIONS:
        if not path.endswith(where):
            continue
        at = text.find(start)
        if at < 0:
            continue
        stop = text.find(end, at)
        spans.append((at, len(text) if stop < 0 else stop + len(end)))
    return spans


def rewrite(text, path):
    """Rename every target, macro and generated header named in the text.

    Spans listed in SKIP_REGIONS are restored afterwards: they name the old
    targets deliberately.

    `kwiver::<target>` keeps its old spelling: that name is the shim, and the
    alias line beside each library is what makes it mean the new target.
    """
    original = text

    # A comment is prose. "The imported vital's headers" and "vital plugin
    # manager" are about kwiver's tree, not about a target, and no rule can
    # tell them apart -- so in a comment only the unambiguous things are
    # renamed: the generated headers and the macros, which are spelled exactly
    # one way. Bare target names in comments are left to a person.
    def _rename(line, comment):
        for old, new in RULES:
            if comment and not (old.endswith("_export.h") or old.isupper()):
                continue
            if old.endswith((".h", ".h.in")):
                head, tail = r"(?<![\w:.])", r"(?![\w])"
            else:
                head = r"(?<![\w:./])"
                tail = r"(?![\w/])(?!\.(?:h|hpp|txx|cxx|cpp|in)\b)"
            line = re.sub(head + re.escape(old) + tail, new, line)
        return line

    lines = text.split("\n")
    text = "\n".join(
        _rename(line, bool(re.match(r"\s*(#|//|\*|/\*)", line))) for line in lines)

    for old, new in ():
        # A whole word, never one that follows `kwiver::` -- that spelling is
        # the shim -- and never one that is really a file name. A target and a
        # source file can share a name: `vital_types` the target and
        # `vital_types.h` the header. The files that should move are in FILES,
        # which is matched first because it is longer.
        if old.endswith((".h", ".h.in")):
            head, tail = r"(?<![\w:.])", r"(?![\w])"
        else:
            # Not a file name, and not a path segment: `kwiver/vital/algo` in a
            # comment is where the code came from, not a target to rename.
            head = r"(?<![\w:./])"
            tail = r"(?![\w/])(?!\.(?:h|hpp|txx|cxx|cpp|in)\b)"
        text = re.sub(head + re.escape(old) + tail, new, text)

    # Put back any directory argument the rules above renamed.
    def _restore(match):
        for was, now in TARGETS.items():
            if match.group(2) == now:
                return match.group(1) + was + match.group(3)
        return match.group(0)

    text = SUBDIR_CALL.sub(_restore, text)

    # And any `python-<path>-<name>` target name, which follows the directory.
    def _restore_derived(match):
        name = match.group(0)
        for was, now in TARGETS.items():
            name = re.sub(r"(?<=python-)" + re.escape(now) + r"(?=[.-])", was, name)
        return name

    text = DERIVED_TARGET.sub(_restore_derived, text)

    # And the module path a python module installs under.
    def _restore_module_path(match):
        head, arg = match.group(1), match.group(2)
        back = {now: was for was, now in TARGETS.items()}
        return head + "/".join(back.get(part, part) for part in arg.split("/"))

    text = PYTHON_MODULE_CALL.sub(_restore_module_path, text)

    # Put back whatever the rules took out of a protected span.
    for start, stop in protected(original, path):
        text = text.replace(rewrite_span(original[start:stop]), original[start:stop])

    return text


def rewrite_span(span):
    """What the rules would do to a span, so it can be put back."""
    out = span
    for old, new in RULES:
        if old.endswith((".h", ".h.in")):
            head, tail = r"(?<![\w:.])", r"(?![\w])"
        else:
            head = r"(?<![\w:./])"
            tail = r"(?![\w/])(?!\.(?:h|hpp|txx|cxx|cpp|in)\b)"
        out = re.sub(head + re.escape(old) + tail, new, out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--check", action="store_true",
                        help="report what would change, write nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    files = tracked(args.paths)
    touched = collections.Counter()

    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            continue

        new_text = rewrite(text, path)
        if new_text == text:
            continue

        changes = sum(1 for a, b in zip(text.split("\n"), new_text.split("\n"))
                      if a != b)
        touched[path] = changes
        if args.verbose:
            print("{:<70} {} line(s)".format(path, changes))
        if not args.check:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(new_text)

    print("\n{} of {} files {}, {} lines".format(
        len(touched), len(files),
        "would change" if args.check else "changed", sum(touched.values())))

    # The files themselves, which a rename of their contents does not move.
    for old, new in FILES.items():
        for found in subprocess.run(["git", "ls-files", "*" + old],
                                    capture_output=True, text=True).stdout.split():
            if os.path.basename(found) != old:
                continue
            target = os.path.join(os.path.dirname(found), new)
            print("  file: {} -> {}".format(found, target))
            if not args.check:
                subprocess.run(["git", "mv", found, target], check=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
