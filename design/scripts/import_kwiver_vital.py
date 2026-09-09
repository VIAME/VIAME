#!/usr/bin/env python3
"""Copy the used part of kwiver's `vital` into `library/`.

Phase 5's second task. The files are the ones `kwiver_reachable.py` found;
this puts each where `lite-library-layout.md` section 2 says it goes and
rewrites the include paths inside it to match. The code itself is not
touched: phases 6 to 8 rewrite it, this only moves it.

Destinations:

    vital/types/*, vital/vital_types.h   -> library/core_types/
    the top-level vital headers that
      vital/types includes               -> library/core_types/
    vital/{config,logger,exceptions,
      util,range,io,plugin_management,
      algo,applets}/                     -> library/algorithm_framework/<sub>/
    the remaining top-level vital files  -> library/algorithm_framework/
    vital/internal/cereal/               -> third_party/cereal/cereal/
      its external/rapidjson             -> third_party/rapidjson/rapidjson/
    vital/kwiversys/                     -> third_party/kwiversys/
    vital/applets/cxxopts.hpp            -> third_party/cxxopts/cxxopts.hpp

Includes are rewritten the same way, so `<vital/types/image.h>` becomes
`<viame/core_types/image.h>` everywhere, in the copied files and in VIAME's
own sources.

The generated headers -- the export headers, `vital_config.h`, `version.h`
and `kwiver-include-paths.h` -- are not copied. `library/algorithm_framework`
generates them, from kwiver's own templates and with kwiver's macro and guard
names, into the build tree where they belong; one of them carries absolute
paths from this build and has no business in the source tree.

Keeping the guard names matters: kwiver's sprokit and arrows headers still
name the old paths, so a translation unit can reach both copies of a header,
and identical guards are what make that one definition rather than two.

Usage:
    import_kwiver_vital.py [--dry-run] [--sources-only]
"""

import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KWIVER = os.path.join(ROOT, "packages", "kwiver")

LIST = os.path.join(ROOT, "design", "lite-kwiver-files.txt")

INCLUDE = re.compile(r'(^\s*#\s*include\s*[<"])([^">]+)([">])', re.MULTILINE)

# Top-level `vital/*.h` that `vital/types/*` includes, so they belong with
# the types rather than with the framework.
CORE_TOP = (
    "vital_types.h", "any.h", "attribute_set.h", "bitflags.h", "context.h",
    "cpp_magic.h", "iterator.h", "math_constants.h", "noncopyable.h",
    "set.h", "signal.h",
)

# Subdirectories of `vital/` that become subdirectories of
# `algorithm_framework/`. `plugin_management` is shortened to `plugin`, as
# lite-library-layout.md section 1 names it.
FRAMEWORK_SUBS = {
    "config": "config",
    "logger": "logger",
    "exceptions": "exceptions",
    "util": "util",
    "range": "range",
    "io": "io",
    "plugin_management": "plugin",
    "algo": "algo",
    "applets": "applets",
}

# Whole directories that move to third_party as they are.
VENDORED = (
    ("vital/kwiversys", "third_party/kwiversys"),
)

# VIAME's own sources whose includes are rewritten alongside the copy.
VIAME_SOURCES = ("library", "plugins", "tools", "examples")

SOURCE_SUFFIXES = (".h", ".hpp", ".hxx", ".txx", ".cxx", ".cpp", ".c", ".cc")


def destination(rel):
    """Where a `vital/...` path goes, relative to the repository root."""
    parts = rel.split("/")

    if parts[0] != "vital":
        return None

    if len(parts) == 2:
        if parts[1] in CORE_TOP:
            return os.path.join("library", "core_types", parts[1])
        return os.path.join("library", "algorithm_framework", parts[1])

    if parts[1] == "types":
        return os.path.join("library", "core_types", *parts[2:])

    if parts[1] == "internal" and parts[2] == "cereal":
        rest = parts[3:]

        if rest[:2] == ["external", "rapidjson"]:
            return os.path.join("third_party", "rapidjson", "rapidjson",
                                *rest[2:])

        return os.path.join("third_party", "cereal", "cereal", *rest)

    if rel == "vital/applets/cxxopts.hpp":
        return os.path.join("third_party", "cxxopts", "cxxopts.hpp")

    if parts[1] in FRAMEWORK_SUBS:
        return os.path.join("library", "algorithm_framework",
                            FRAMEWORK_SUBS[parts[1]], *parts[2:])

    return None


def rewritten(spec):
    """What an include of a kwiver path becomes, or None to leave it alone."""
    if spec.startswith("vital/internal/cereal/external/rapidjson/"):
        return "rapidjson/" + spec[len(
            "vital/internal/cereal/external/rapidjson/"):]

    if spec.startswith("vital/internal/cereal/"):
        return "cereal/" + spec[len("vital/internal/cereal/"):]

    if spec == "vital/applets/cxxopts.hpp":
        return "cxxopts.hpp"

    if not spec.startswith("vital/"):
        return None

    target = destination(spec)

    if target is None:
        return None

    parts = target.split(os.sep)

    if parts[0] == "third_party":
        return "/".join(parts[2:])

    # library/<name>/<rest> -> viame/<name>/<rest>
    return "/".join(["viame"] + parts[1:])


# A generated export header included by its bare name, resolved against the
# including file's own directory. That no longer works once the header is
# generated into its own root, so it is spelled out like the rest.
BARE_EXPORT = re.compile(r"^vital_[a-z_]*export\.h$")


def rewrite_text(text, here=None):
    def replace(match):
        spec = match.group(2)
        target = rewritten(spec)

        if target is None and here and BARE_EXPORT.match(spec):
            target = here + "/" + spec

        if target is None:
            return match.group(0)

        return match.group(1) + target + match.group(3)

    return INCLUDE.sub(replace, text)


def listed_vital_files():
    with open(LIST) as handle:
        paths = [line.strip() for line in handle if line.strip()]

    prefix = os.path.join("packages", "kwiver") + os.sep

    return [path[len(prefix):] for path in paths
            if path.startswith(prefix + "vital" + os.sep)]


def copy(source, target, dry_run, here=None):
    if dry_run:
        return

    os.makedirs(os.path.dirname(target), exist_ok=True)

    with open(source, errors="surrogateescape") as handle:
        text = handle.read()

    with open(target, "w", errors="surrogateescape") as handle:
        handle.write(rewrite_text(text, here))


# Cleared before each copy, so a file that stops being reachable stops being
# here. Anything not written by the copy -- the CMakeLists, in particular --
# is left alone.
COPY_TARGETS = (
    os.path.join("library", "core_types"),
    os.path.join("library", "algorithm_framework"),
)


def clear(dry_run):
    removed = 0

    for directory in COPY_TARGETS:
        base = os.path.join(ROOT, directory)

        if not os.path.isdir(base):
            continue

        for walk_base, dirs, names in os.walk(base, topdown=False):
            for name in names:
                if not name.endswith(SOURCE_SUFFIXES):
                    continue

                removed += 1

                if not dry_run:
                    os.remove(os.path.join(walk_base, name))

            if not dry_run and not os.listdir(walk_base) and walk_base != base:
                os.rmdir(walk_base)

    return removed


def copy_sources(dry_run):
    copied, skipped = 0, []

    for rel in listed_vital_files():
        target = destination(rel)

        if target is None:
            skipped.append(rel)
            continue

        parts = os.path.dirname(target).split(os.sep)
        here = "/".join(["viame"] + parts[1:]) if parts[0] == "library" else None

        copy(os.path.join(KWIVER, rel), os.path.join(ROOT, target), dry_run,
             here)
        copied += 1

    return copied, skipped


def copy_vendored(dry_run):
    copied = 0

    for source_rel, target_rel in VENDORED:
        source = os.path.join(KWIVER, source_rel)
        target = os.path.join(ROOT, target_rel)

        if not os.path.isdir(source):
            continue

        if not dry_run:
            if os.path.isdir(target):
                shutil.rmtree(target)
            shutil.copytree(source, target)

        copied += sum(len(names) for _, _, names in os.walk(source))

    return copied


def rewrite_viame(dry_run):
    """Point VIAME's own sources at the copy."""
    changed = 0

    for directory in VIAME_SOURCES:
        base = os.path.join(ROOT, directory)

        if not os.path.isdir(base):
            continue

        for walk_base, dirs, names in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]

            for name in sorted(names):
                if not name.endswith(SOURCE_SUFFIXES):
                    continue

                path = os.path.join(walk_base, name)

                with open(path, errors="surrogateescape") as handle:
                    text = handle.read()

                updated = rewrite_text(text)

                if updated == text:
                    continue

                changed += 1

                if not dry_run:
                    with open(path, "w", errors="surrogateescape") as handle:
                        handle.write(updated)

    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sources-only", action="store_true",
                        help="do not rewrite VIAME's own includes")
    args = parser.parse_args()

    if not os.path.isfile(LIST):
        print("no {}; run kwiver_reachable.py first".format(
            os.path.relpath(LIST, ROOT)), file=sys.stderr)
        return 1

    print("{} files cleared from the copy destinations".format(
        clear(args.dry_run)))

    copied, skipped = copy_sources(args.dry_run)
    print("{} vital files copied".format(copied))

    if skipped:
        print("\n{} listed vital files had no destination:".format(
            len(skipped)), file=sys.stderr)
        for rel in skipped[:20]:
            print("  {}".format(rel), file=sys.stderr)

    print("{} vendored files copied to third_party".format(
        copy_vendored(args.dry_run)))

    if not args.sources_only:
        print("{} VIAME sources had their includes rewritten".format(
            rewrite_viame(args.dry_run)))

    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
