#!/usr/bin/env python3
"""Move a group of kwiver's code into `library/`.

Phase 5. Each group is one run: the files go where
`lite-library-layout.md` section 2 says they go, every include of them is
rewritten to match, and kwiver's copies are deleted in the same change so
that only one of each library is ever built. Two copies of identical code
carry two copies of the static state and do not survive teardown; see
`lite-findings.md` section 1.1.

The code itself is not touched. Phases 6 to 8 rewrite it; this only moves it.

Groups:

  vital (P5-T02, done)
    vital/types/*, vital/vital_types.h   -> library/core_types/
    the top-level vital headers that
      vital/types includes               -> library/core_types/
    vital/{config,logger,exceptions,
      util,range,io,plugin_management,
      algo,applets}/                     -> library/algorithm_framework/<sub>/
    the remaining top-level vital files  -> library/algorithm_framework/
    vital/internal/cereal/               -> third_party/cereal/cereal/
      (whole, as upstream ships it, including its bundled rapidjson: it is a
       vendored library, and splitting it would mean patching its includes)
    vital/kwiversys/                     -> third_party/kwiversys/
    vital/applets/cxxopts.hpp            -> third_party/cxxopts/cxxopts.hpp

  sprokit (P5-T03)
    sprokit/src/sprokit/pipeline/*       -> library/pipeline_framework/
    sprokit/src/sprokit/pipeline_util/*  -> library/pipeline_framework/
      (both flatten: the two directories share no file name, and the layout
       lists the engine and the .pipe parser as one thing)
    sprokit/src/schedulers/*             -> library/pipeline_framework/schedulers/
    sprokit/src/applets/*                -> library/pipeline_framework/applets/
    sprokit/processes/adapters/*         -> library/pipeline_framework/adapters/
    sprokit/processes/kwiver_type_traits.h
                                         -> library/pipeline_framework/type_traits.h
    sprokit/processes/trait_utils.h      -> library/pipeline_framework/trait_utils.h
    the four generic processes           -> library/pipeline_framework/processes/

Unlike vital, the sprokit directories come across whole rather than as the
subset `kwiver_reachable.py` found. They are complete libraries that kwiver
builds as units, and the closure's blind spots -- registration by type, most
of all -- are exactly what would be lost. `sprokit/processes/core` is the
one place a subset is taken, and only the processes the layout assigns here;
the rest go to the functional directories in P5-T04.

The generated headers -- the export headers, `vital_config.h`, `version.h`,
`kwiver-include-paths.h` -- are not copied. Kwiver's build generates them,
from its own templates and with its own macro and guard names, into the
build tree where they belong; one of them carries absolute paths from the
build and has no business in the source tree.

A group is one-shot: once kwiver's copies are gone there is nothing left to
read, and running it again would empty the destinations. Each group refuses
to run when its source is already gone.

Usage:
    import_kwiver.py <group> [--dry-run] [--sources-only]
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
    ("vital/internal/cereal", "third_party/cereal/cereal"),
)

# VIAME's own sources whose includes are rewritten alongside the copy.
VIAME_SOURCES = ("library", "plugins", "tools", "examples", "tests")

# What is left of kwiver once vital moves: its arrows, its pipeline engine and
# its python bindings all include vital by the old paths, and there can be
# only one copy of vital in a process, so they are rewritten too. See the note
# on P5-T02 in tasks/phase-05-import-kwiver.md for what happens when there are
# two.
KWIVER_SOURCES = (
    "arrows", "sprokit", "python", "examples", "extras", "tools",
    # What stays behind in `vital/`: the small plugin modules and the tools
    # that are not part of the imported subset but still include it.
    "vital/config_plugins", "vital/logger_plugins", "vital/applets_plugins",
    "vital/tools", "vital/test_interface",
)

SOURCE_SUFFIXES = (".h", ".hpp", ".hxx", ".txx", ".cxx", ".cpp", ".c", ".cc")


# Where each sprokit directory lands under `library/pipeline_framework`.
# `pipeline` and `pipeline_util` both flatten: they share no file name, and
# the layout lists the engine and the .pipe parser as one thing.
SPROKIT_SUBS = {
    "sprokit/src/sprokit/pipeline": "",
    "sprokit/src/sprokit/pipeline_util": "",
    "sprokit/src/schedulers": "schedulers",
    "sprokit/src/applets": "applets",
    "sprokit/processes/adapters": "adapters",
}

# Single files, and the one rename the task asks for.
SPROKIT_FILES = {
    "sprokit/processes/kwiver_type_traits.h":
        "library/pipeline_framework/type_traits.h",
    "sprokit/processes/trait_utils.h":
        "library/pipeline_framework/trait_utils.h",
}

# The generic processes the layout assigns to the pipeline framework. Only
# one of the four it names is kwiver's: `filter_frame`, `filter_frame_index`
# and `image_to_image_set` are VIAME's own, and move with `plugins/core` in
# phase 2.
SPROKIT_PROCESSES = (
    "sprokit/processes/core/downsample_process.h",
    "sprokit/processes/core/downsample_process.cxx",
)


def sprokit_destination(rel):
    """Where a `sprokit/...` path goes, relative to the repository root."""
    if rel in SPROKIT_FILES:
        return SPROKIT_FILES[rel]

    if rel in SPROKIT_PROCESSES:
        return os.path.join("library", "pipeline_framework", "processes",
                            os.path.basename(rel))

    for source, sub in SPROKIT_SUBS.items():
        prefix = source + "/"

        if not rel.startswith(prefix):
            continue

        rest = rel[len(prefix):]

        # Nothing below the directory itself: the tests and the worked
        # examples are not what gets copied.
        if "/" in rest:
            return None

        return os.path.join("library", "pipeline_framework", sub, rest)

    return None


def destination(rel):
    """Where a kwiver path goes, relative to the repository root."""
    parts = rel.split("/")

    if parts[0] == "sprokit":
        return sprokit_destination(rel)

    if parts[0] != "vital":
        return None

    if len(parts) == 2:
        if parts[1] in CORE_TOP:
            return os.path.join("library", "core_types", parts[1])
        return os.path.join("library", "algorithm_framework", parts[1])

    if parts[1] == "types":
        return os.path.join("library", "core_types", *parts[2:])

    if parts[1] == "internal" and parts[2] == "cereal":
        # Vendored whole by VENDORED below rather than file by file
        return None

    if rel == "vital/applets/cxxopts.hpp":
        return os.path.join("third_party", "cxxopts", "cxxopts.hpp")

    if parts[1] in FRAMEWORK_SUBS:
        return os.path.join("library", "algorithm_framework",
                            FRAMEWORK_SUBS[parts[1]], *parts[2:])

    return None


def rewritten(spec):
    """What an include of a kwiver path becomes, or None to leave it alone."""
    if spec.startswith("vital/internal/cereal/"):
        return "cereal/" + spec[len("vital/internal/cereal/"):]

    if spec == "vital/applets/cxxopts.hpp":
        return "cxxopts.hpp"

    if spec.startswith("sprokit/"):
        # An include names the header the way kwiver's include path reaches
        # it -- `sprokit/pipeline/x.h`, not `sprokit/src/sprokit/pipeline/x.h`
        # -- so put the source root back before looking it up.
        for candidate in ("sprokit/src/" + spec, spec):
            target = destination(candidate)

            if target is not None:
                return "/".join(["viame"] + target.split(os.sep)[1:])

        return None

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
BARE_EXPORT = re.compile(r"^(vital|sprokit|kwiver)_[a-z_0-9]*export\.h$")

# Bare names that kwiver reached through a directory on its include path
# rather than through a path of their own.
BARE_MOVED = {
    "kwiver_type_traits.h": "viame/pipeline_framework/type_traits.h",
    "trait_utils.h": "viame/pipeline_framework/trait_utils.h",
}


def rewrite_text(text, here=None):
    def replace(match):
        spec = match.group(2)
        target = rewritten(spec)

        if target is None:
            target = BARE_MOVED.get(spec)

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


def clear(targets, dry_run):
    """Empty a group's destinations before copying into them.

    A file that stops being part of the group stops being here. Anything the
    copy does not write -- the CMakeLists, in particular -- is left alone.
    """
    removed = 0

    for directory in targets:
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


def copy_sources(sources, dry_run):
    copied, skipped = 0, []

    for rel in sources:
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


def copy_vendored(vendored, dry_run):
    copied = 0

    for source_rel, target_rel in vendored:
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


def rewrite_tree(root, directories, dry_run):
    """Point a tree's includes at the copy."""
    changed = 0

    for directory in directories:
        base = os.path.join(root, directory)

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


def sprokit_sources():
    """Every file the sprokit group moves, kwiver-relative.

    Whole directories, unlike vital: these are complete libraries that
    kwiver builds as units, and what a closure over includes cannot see --
    registration by type, most of all -- is exactly what would be lost. See
    `lite-findings.md` section 1.4.
    """
    found = list(SPROKIT_FILES) + list(SPROKIT_PROCESSES)

    for source in SPROKIT_SUBS:
        base = os.path.join(KWIVER, source)

        if not os.path.isdir(base):
            continue

        for name in sorted(os.listdir(base)):
            if name.endswith(SOURCE_SUFFIXES):
                found.append(source + "/" + name)

    return [rel for rel in found
            if os.path.isfile(os.path.join(KWIVER, rel))]


GROUPS = {
    "vital": {
        "task": "P5-T02",
        # Gone once the group has landed, so the run is refused rather than
        # emptying the destinations
        "sentinel": "vital/types/image.h",
        "sources": listed_vital_files,
        "clear": (os.path.join("library", "core_types"),
                  os.path.join("library", "algorithm_framework")),
        "vendored": VENDORED,
    },
    "sprokit": {
        "task": "P5-T03",
        "sentinel": "sprokit/src/sprokit/pipeline/process.h",
        "sources": sprokit_sources,
        "clear": (os.path.join("library", "pipeline_framework"),),
        "vendored": (),
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=sorted(GROUPS),
                        help="which group of kwiver code to move")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sources-only", action="store_true",
                        help="do not rewrite VIAME's own includes")
    args = parser.parse_args()

    group = GROUPS[args.group]

    if not os.path.isfile(LIST):
        print("no {}; run kwiver_reachable.py first".format(
            os.path.relpath(LIST, ROOT)), file=sys.stderr)
        return 1

    if not os.path.isfile(os.path.join(KWIVER, group["sentinel"])):
        print("kwiver no longer has the {} sources: {} has landed and "
              "library/ is the source of truth now. Running this again would "
              "empty the destinations.".format(args.group, group["task"]),
              file=sys.stderr)
        return 1

    print("{} files cleared from the copy destinations".format(
        clear(group["clear"], args.dry_run)))

    copied, skipped = copy_sources(group["sources"](), args.dry_run)
    print("{} {} files copied".format(copied, args.group))

    if skipped:
        print("\n{} listed files had no destination:".format(
            len(skipped)), file=sys.stderr)
        for rel in skipped[:20]:
            print("  {}".format(rel), file=sys.stderr)

    if group["vendored"]:
        print("{} vendored files copied to third_party".format(
            copy_vendored(group["vendored"], args.dry_run)))

    if not args.sources_only:
        print("{} VIAME sources had their includes rewritten".format(
            rewrite_tree(ROOT, VIAME_SOURCES, args.dry_run)))
        print("{} kwiver sources had their includes rewritten".format(
            rewrite_tree(KWIVER, KWIVER_SOURCES, args.dry_run)))

    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
