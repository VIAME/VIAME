#!/usr/bin/env python3
"""Which kwiver source files VIAME actually reaches.

Phase 5 copies the used part of kwiver into `library/`. This works out what
that part is, so the copy is a list rather than a judgement call.

It starts from three places and follows `#include` lines from each until the
set stops growing:

* every kwiver header included from VIAME's own C++ (`library/`, `plugins/`
  and `tools/`), which is what VIAME compiles against;
* every implementation named in `tests/baseline/registry.json` and not in
  `removed.json`, found by looking for its name in a registration file and
  taking the headers that file includes. Those are reached at run time
  through the plugin loader rather than at compile time, so nothing else
  would find them;
* the sprokit engine, the pipeline runner and the adapters, which are how a
  pipeline runs at all and which nothing includes by name.

Tests and worked examples are dropped from the result: they are not what
gets copied, and anything they use that is really used has come in through
some other path already.

kwiversys is absent for a different reason. Its headers are `.h.in` and
`.hxx.in` templates configured into the build directory, so an include of
`<kwiversys/SystemTools.hxx>` resolves to nothing in the source tree. It is
used, and lite-library-layout.md moves the whole directory to
`third_party/kwiversys`; there is nothing for a closure over source files to
discover about it.

A header brings its implementation with it: `X.h` pulls in `X.cxx` where
there is one, since the point of the list is what to copy.

Usage:
    kwiver_reachable.py [--kwiver DIR] [--output FILE] [--verbose]

Writes `design/lite-kwiver-files.txt`, one repository-relative path per line,
sorted, and prints the line count and a per-directory summary.
"""

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]')

# Where VIAME's own C++ lives. `plugins/` is here because phases 1 and 2 are
# deferred: most of VIAME is still there rather than in `library/`.
VIAME_SOURCES = ("library", "plugins", "tools")

SOURCE_SUFFIXES = (".h", ".hpp", ".hxx", ".txx", ".cxx", ".cpp", ".c", ".cc")
HEADER_SUFFIXES = (".h", ".hpp", ".hxx", ".txx")

# Directories a `#include <...>` is resolved against, relative to the kwiver
# root, mirroring the include path kwiver's own targets are built with.
INCLUDE_ROOTS = ("", "sprokit/src", "sprokit/processes")

# Reached only through the plugin loader and the scheduler, so no include
# points at them.
ENTRY_POINTS = (
    "sprokit/src/sprokit/pipeline",
    "sprokit/src/sprokit/pipeline_util",
    "sprokit/src/schedulers",
    "sprokit/src/applets/pipeline_runner.cxx",
    "sprokit/processes/adapters",
    "vital/applets",
    "vital/plugin_management",
)

# Which file a name of each kind is looked up in. Matching a name against
# every registration file finds it in the wrong one: the applet named `json`
# matches `arrows/serialize/json/algo/register_algorithms.cxx`, which pulls
# in a whole arrow that registers nothing VIAME uses.
REGISTRATION_FILES = {
    "algorithm": ("register_algorithms.cxx", "register_factories.cxx"),
    "process": ("register_processes.cxx", "register_factories.cxx"),
    "applet": ("register_applets.cxx", "register_factories.cxx"),
    "scheduler": ("register_schedulers.cxx", "register_factories.cxx"),
    "cluster": (),
}

# Arrows phases 3 and 4 turned off. Nothing in them is built, so nothing may
# be reached through them either; a registration file in here still names
# implementations that `registry.json` records, and following those would
# walk straight back into the dependency that was just removed.
NOT_BUILT = ("arrows/vxl", "arrows/ffmpeg")

# Nothing under these may appear in the result; phase 5 lists them as the
# check that the closure has not wandered.
FORBIDDEN = (
    "arrows/klv", "arrows/serialize", "arrows/dbow2", "arrows/vtk",
    "arrows/kpf", "arrows/ceres", "arrows/qt", "arrows/super3d",
    "arrows/geocalc", "arrows/gdal", "arrows/cuda", "arrows/zlib",
    "arrows/pdal", "arrows/proj", "arrows/uuid", "arrows/matlab",
)


def is_built(kwiver, path):
    return not any(
        path.startswith(os.path.join(kwiver, directory) + os.sep)
        for directory in NOT_BUILT)


def kwiver_root(argument):
    return os.path.abspath(argument or os.path.join(ROOT, "packages", "kwiver"))


def walk(directory, suffixes):
    for base, dirs, names in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__")]

        for name in sorted(names):
            if name.endswith(suffixes):
                yield os.path.join(base, name)


def includes_of(path):
    try:
        with open(path, errors="replace") as handle:
            text = handle.read()
    except OSError:
        return []

    return [match.group(1) for match in INCLUDE.finditer(text)
            if INCLUDE.match(match.group(0))]


def read_includes(path):
    found = []

    try:
        with open(path, errors="replace") as handle:
            for line in handle:
                match = INCLUDE.match(line)
                if match:
                    found.append(match.group(1))
    except OSError:
        pass

    return found


def resolve(kwiver, spec, origin):
    """The kwiver file an include names, or None if it names something else.

    Tried in the order the compiler would: beside the including file, then
    each include root. A spec that resolves nowhere is either a system
    header, a generated one such as an export header, or a dependency's.
    """
    candidates = []

    if origin:
        candidates.append(os.path.normpath(
            os.path.join(os.path.dirname(origin), spec)))

    for root in INCLUDE_ROOTS:
        candidates.append(os.path.normpath(os.path.join(kwiver, root, spec)))

    for candidate in candidates:
        if (candidate.startswith(kwiver) and os.path.isfile(candidate)
                and is_built(kwiver, candidate)):
            return candidate

    return None


def companions(path):
    """The implementation files that belong with a header."""
    if not path.endswith(HEADER_SUFFIXES):
        return []

    stem = os.path.splitext(path)[0]

    return [stem + suffix for suffix in (".cxx", ".cpp", ".c", ".txx")
            if os.path.isfile(stem + suffix)]


def mentions(text, kind, name):
    """Whether a registration file registers `name`.

    Algorithms and applets are registered under a quoted name. Processes are
    registered by type -- `reg.register_process< frame_list_process >()` --
    with the name coming off the class, so the only thing in the file to
    match is the type, which is the name with `_process` on the end.
    """
    if '"{}"'.format(name) in text:
        return True

    return kind == "process" and "{}_process".format(name) in text


def registration_roots(kwiver, verbose):
    """Files that register a name VIAME still uses, and the headers they use.

    A plugin is reached by name at run time, so the registration file is the
    only thing that connects the name to the code behind it.
    """
    with open(os.path.join(ROOT, "tests", "baseline", "registry.json")) as h:
        registry = json.load(h)

    with open(os.path.join(ROOT, "tests", "baseline", "removed.json")) as h:
        removed = {(entry["kind"], entry.get("interface", ""), entry["name"])
                   for entry in json.load(h)}

    wanted = collections.defaultdict(set)

    for interface, entries in sorted(registry.get("algorithms", {}).items()):
        for name in entries:
            if ("algorithm", interface, name) not in removed:
                wanted["algorithm"].add(name)

    for plural, kind in (("processes", "process"), ("applets", "applet"),
                         ("clusters", "cluster"),
                         ("schedulers", "scheduler")):
        for name in registry.get(plural, {}):
            if (kind, "", name) not in removed:
                wanted[kind].add(name)

    roots, matched = set(), set()

    for path in walk(kwiver, (".cxx",)):
        base = os.path.basename(path)

        if not is_built(kwiver, path):
            continue

        kinds = [kind for kind, files in REGISTRATION_FILES.items()
                 if base in files]

        if not kinds:
            continue

        try:
            with open(path, errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue

        here = {name for kind in kinds for name in wanted[kind]
                if mentions(text, kind, name)}

        if here:
            roots.add(path)
            matched |= here

    if verbose:
        every = set().union(*wanted.values()) if wanted else set()
        missing = sorted(every - matched)
        print("registration names not found in any kwiver registration file: "
              "{} (these are VIAME's own, or python)".format(len(missing)),
              file=sys.stderr)
        for name in missing[:40]:
            print("  {}".format(name), file=sys.stderr)

    return roots


def entry_point_roots(kwiver):
    roots = set()

    for entry in ENTRY_POINTS:
        path = os.path.join(kwiver, entry)

        if os.path.isfile(path):
            roots.add(path)
        elif os.path.isdir(path):
            roots.update(p for p in walk(path, SOURCE_SUFFIXES)
                         if "/tests/" not in p and "/examples/" not in p)

    return roots


def viame_roots(kwiver):
    """Kwiver files VIAME's own C++ includes directly."""
    roots = set()

    for directory in VIAME_SOURCES:
        base = os.path.join(ROOT, directory)

        if not os.path.isdir(base):
            continue

        for path in walk(base, SOURCE_SUFFIXES):
            for spec in read_includes(path):
                target = resolve(kwiver, spec, origin=None)

                if target:
                    roots.add(target)

    return roots


def close(kwiver, roots, verbose):
    """Everything reachable from `roots`, and what first reached each thing.

    The second half matters as much as the first: when a directory turns up
    that should not be in the result, the only useful question is which
    include put it there, and that is what the reached-by chain answers.
    """
    reached_by = {path: None for path in roots}
    seen = set()
    queue = collections.deque(roots)
    unresolved = collections.Counter()

    while queue:
        path = queue.popleft()

        if path in seen or not os.path.isfile(path):
            continue

        seen.add(path)

        for companion in companions(path):
            if companion not in seen:
                reached_by.setdefault(companion, path)
                queue.append(companion)

        for spec in read_includes(path):
            target = resolve(kwiver, spec, origin=path)

            if target:
                if target not in seen:
                    reached_by.setdefault(target, path)
                    queue.append(target)
            elif "/" in spec:
                unresolved[spec] += 1

    if verbose:
        print("\nincludes that resolve outside kwiver (system, generated or "
              "a dependency), most common first:", file=sys.stderr)
        for spec, count in unresolved.most_common(30):
            print("  {:5d}  {}".format(count, spec), file=sys.stderr)

    return seen, reached_by


def chain_to(reached_by, path, limit=8):
    """How the closure got to `path`, nearest first."""
    steps, seen = [], set()

    while path and path not in seen and len(steps) < limit:
        seen.add(path)
        steps.append(path)
        path = reached_by.get(path)

    return steps


def lines_in(path):
    try:
        with open(path, errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kwiver", default=None,
                        help="kwiver source directory")
    parser.add_argument("--output", default=os.path.join(
        ROOT, "design", "lite-kwiver-files.txt"))
    parser.add_argument("--verbose", action="store_true",
                        help="report unresolved includes and unmatched names")
    args = parser.parse_args()

    kwiver = kwiver_root(args.kwiver)

    if not os.path.isdir(kwiver):
        print("no kwiver at {}".format(kwiver), file=sys.stderr)
        return 1

    roots = viame_roots(kwiver)
    roots |= registration_roots(kwiver, args.verbose)
    roots |= entry_point_roots(kwiver)

    reached, reached_by = close(kwiver, roots, args.verbose)

    # Tests are not part of what gets copied; a test that includes a header
    # has already brought that header in through some other path if it is
    # really used
    reached = {path for path in reached
               if "/tests/" not in path and "/examples/" not in path}

    relative = sorted(os.path.relpath(path, ROOT) for path in reached)

    with open(args.output, "w") as handle:
        handle.write("\n".join(relative) + "\n")

    total = sum(lines_in(os.path.join(ROOT, path)) for path in relative)

    print("{} files, {} lines -> {}".format(
        len(relative), total, os.path.relpath(args.output, ROOT)))

    groups = collections.Counter()

    for path in relative:
        parts = path.split(os.sep)
        groups[os.sep.join(parts[2:4])] += 1

    print("\nby directory:")
    for group, count in sorted(groups.items()):
        print("  {:5d}  {}".format(count, group))

    failures = 0

    for directory in FORBIDDEN:
        inside = sorted(path for path in reached
                        if os.path.join(kwiver, directory) + os.sep in
                        path + os.sep)

        if not inside:
            continue

        failures += 1
        print("\n{} is reached ({} files); the shortest way in:".format(
            directory, len(inside)), file=sys.stderr)

        shortest = min(
            (chain_to(reached_by, path) for path in inside), key=len)

        for step in shortest:
            print("    {}".format(os.path.relpath(step, ROOT)),
                  file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
