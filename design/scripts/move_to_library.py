#!/usr/bin/env python3
"""Move an arrow's classes or a process out of kwiver into a VIAME library.

P5-T04 splits `arrows/core`, `arrows/ocv`, `arrows/mvg` and
`sprokit/processes/core` across the functional libraries. Each destination is
one run of this, which does the mechanical half:

* copies the files to `library/<destination>/`, rewriting their include paths
  and their export header and macro -- an arrow's class cannot keep saying
  `KWIVER_ALGO_OCV_EXPORT` once it is built somewhere else;
* deletes kwiver's copies;
* takes the entries out of the arrow's class list or the process directory's
  source list, which name files without extensions, so a prune that matches
  `<name>.h` matches nothing and the error arrives later as "No SOURCES given
  to target";
* takes the registration out of kwiver's `register_algorithms.cxx` or
  `register_processes.cxx`, by statement rather than by line, so a factory
  call and the attribute chain that follows it go together.

What it does not do is write the destination's own CMakeLists and
registration. Those differ per library and are worth reading.

The namespace is left alone. Phase 8 is where that changes; changing it here
would mean touching every file for no gain while the code is still moving.

Usage:
    move_to_library.py <destination> <kwiver-path> [<kwiver-path> ...]
                       [--process] [--dry-run]

A kwiver path is given without its extension, as the class lists spell it:

    move_to_library.py file_io arrows/core/algo/detected_object_set_input_kw18
    move_to_library.py file_io sprokit/processes/core/read_object_track_process \\
        --process
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KWIVER = os.path.join(ROOT, "packages", "kwiver")

sys.path.insert(0, HERE)
import import_kwiver as imp                                    # noqa: E402

SUFFIXES = (".h", ".cxx", ".txx")

# The export header a moved file gets, by whether it is a process.
EXPORT_MACROS = re.compile(
    r"\bKWIVER_(?:ALGO_[A-Z]+|PROCESSES(?:_[A-Z]+)?)_"
    r"(NO_EXPORT|EXPORT|DEPRECATED[A-Z_]*)\b")

EXPORT_INCLUDE = re.compile(
    r'#include\s*[<"][^">]*kwiver_[a-z_0-9]*export\.h[">]')


def destination_names(library, is_process):
    if is_process:
        return ("viame_processes_{}_export.h".format(library),
                "VIAME_PROCESSES_{}".format(library.upper()))
    return ("viame_{}_export.h".format(library),
            "VIAME_{}".format(library.upper()))


def move(library, rel, is_process, dry_run):
    """Copy one class's files across and delete kwiver's."""
    header, macro = destination_names(library, is_process)
    moved = []

    for suffix in SUFFIXES:
        source = os.path.join(KWIVER, rel + suffix)

        if not os.path.isfile(source):
            continue

        name = os.path.basename(rel) + suffix
        target = os.path.join(ROOT, "library", library, name)

        text = open(source, errors="surrogateescape").read()
        text = imp.rewrite_text(text, "viame/" + library)
        text = EXPORT_INCLUDE.sub('#include "{}"'.format(header), text)
        text = EXPORT_MACROS.sub(
            lambda m: "{}_{}".format(macro, m.group(1)), text)

        # A file that included its own header by name keeps working
        text = text.replace('#include "{}.h"'.format(os.path.basename(rel)),
                            '#include "{}.h"'.format(os.path.basename(rel)))

        if not dry_run:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            open(target, "w", errors="surrogateescape").write(text)
            os.remove(source)

        moved.append(name)

    return moved


def prune_lists(rel, dry_run):
    """Take the entry out of whatever CMakeLists names it.

    Class lists spell an entry without its extension and relative to the
    directory's own CMakeLists, source lists spell it with one, so both
    spellings are tried.
    """
    directory = os.path.dirname(rel)
    stem = os.path.basename(rel)
    pruned = []

    # An arrow's class list lives one directory up from `algo/`
    candidates = [os.path.join(KWIVER, directory, "CMakeLists.txt"),
                  os.path.join(KWIVER, os.path.dirname(directory),
                               "CMakeLists.txt")]

    spellings = {stem, os.path.basename(directory) + "/" + stem,
                 stem + ".h", stem + ".cxx", stem + ".txx"}

    for path in candidates:
        if not os.path.isfile(path):
            continue

        text = open(path).read()
        out = [line for line in text.splitlines(keepends=True)
               if line.strip() not in spellings]

        if "".join(out) != text:
            if not dry_run:
                open(path, "w").write("".join(out))
            pruned.append(os.path.relpath(path, KWIVER))

    return pruned


REGISTRATION = (
    r'(?:auto\s+)?fact\s*=\s*vp[ml]\.add_factory<[^;]*?\b{}\b[^;]*?>\s*'
    r'\(\s*"[^"]+"\s*\)\s*;(?:\s*fact->[^;]*;)*\s*')


def deregister(rel, is_process, dry_run):
    """Take the registration and the include out of kwiver's register file."""
    directory = os.path.dirname(rel)
    stem = os.path.basename(rel)
    name = "register_processes.cxx" if is_process else "register_algorithms.cxx"

    for path in (os.path.join(KWIVER, directory, name),
                 os.path.join(KWIVER, os.path.dirname(directory), name)):
        if not os.path.isfile(path):
            continue

        text = open(path).read()
        original = text

        if is_process:
            text = "".join(
                line for line in text.splitlines(keepends=True)
                if "register_process< {} >".format(stem) not in line
                and '#include "{}.h"'.format(stem) not in line)
        else:
            text = re.sub(REGISTRATION.format(re.escape(stem)), "", text,
                          flags=re.S)
            text = "".join(
                line for line in text.splitlines(keepends=True)
                if not re.search(r"#include.*\b{}\.h".format(re.escape(stem)),
                                 line))
            if "auto fact" not in text:
                text = re.sub(r"(\n\s*)fact = vp([ml])\.add_factory",
                              r"\1auto fact = vp\2.add_factory", text, count=1)

        if text != original:
            if not dry_run:
                open(path, "w").write(text)
            return os.path.relpath(path, KWIVER)

    return None


# Where each moved file went, keyed by the kwiver path an include names it
# by. Recorded rather than guessed: keying on the base name alone rewrote
# every `arrows/ocv/image_container.h` to `viame/core_types/image_container.h`,
# because arrows/ocv's vital bridge and core_types share half a dozen file
# names.
LEDGER = os.path.join(ROOT, "design", "lite-kwiver-moves.txt")


def read_ledger():
    moves = {}

    if os.path.isfile(LEDGER):
        for line in open(LEDGER):
            line = line.strip()

            if line and not line.startswith("#"):
                source, target = line.split()
                moves[source] = target

    return moves


def write_ledger(moves):
    with open(LEDGER, "w") as handle:
        handle.write(
            "# Where P5-T04 moved each kwiver file, as an include names it on\n"
            "# the left and as it is named now on the right. Written by\n"
            "# design/scripts/move_to_library.py; the record of the split.\n")

        for source in sorted(moves):
            handle.write("{} {}\n".format(source, moves[source]))


def include_spellings(rel, name):
    """The paths an include could have named a moved file by.

    Kwiver reaches `arrows/ocv/algo/x.h` as itself and, from a sibling, as
    `x.h`; the second is left alone, because a sibling that moved with it
    still finds it beside itself.
    """
    spellings = {rel}

    if "/algo/" in rel:
        spellings.add(rel.replace("/algo/", "/"))

    return spellings


def rewrite_moved_includes(dry_run):
    """Point every include of a moved file at where it went."""
    moves = read_ledger()

    if not moves:
        return 0

    pattern = re.compile(
        r'(#\s*include\s*[<"])(' +
        "|".join(re.escape(k) for k in sorted(moves, key=len, reverse=True)) +
        r')([">])')

    changed = 0

    for root, directories in ((ROOT, ("library", "plugins", "tools",
                                      "examples", "tests")),
                              (KWIVER, ("arrows", "sprokit", "python",
                                        "vital", "tools"))):
        for directory in directories:
            base = os.path.join(root, directory)

            if not os.path.isdir(base):
                continue

            for walk, dirs, names in os.walk(base):
                dirs[:] = [d for d in dirs if d != "__pycache__"]

                for name in sorted(names):
                    if not name.endswith(imp.SOURCE_SUFFIXES):
                        continue

                    path = os.path.join(walk, name)
                    text = open(path, errors="surrogateescape").read()
                    out = pattern.sub(
                        lambda m: m.group(1) + moves[m.group(2)] + m.group(3),
                        text)

                    if out != text:
                        changed += 1
                        if not dry_run:
                            open(path, "w",
                                 errors="surrogateescape").write(out)

    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", help="destination under library/")
    parser.add_argument("paths", nargs="+",
                        help="kwiver paths without an extension")
    parser.add_argument("--process", action="store_true",
                        help="these are sprokit processes, not arrow classes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    moves = read_ledger()

    for rel in args.paths:
        moved = move(args.library, rel, args.process, args.dry_run)

        if not moved:
            print("nothing to move for {}".format(rel), file=sys.stderr)
            continue

        pruned = prune_lists(rel, args.dry_run)
        where = deregister(rel, args.process, args.dry_run)

        for name in moved:
            if not name.endswith((".h", ".txx")):
                continue

            for spelling in include_spellings(
                    os.path.dirname(rel) + "/" + name, name):
                moves[spelling] = "viame/{}/{}".format(args.library, name)

        print("{:52s} -> library/{}  [{}]".format(
            rel, args.library, ", ".join(moved)))

        if not pruned:
            print("   no CMakeLists names it", file=sys.stderr)
        if where is None:
            print("   no registration found", file=sys.stderr)

    if not args.dry_run:
        write_ledger(moves)

    print("{} files had an include of a moved header rewritten".format(
        rewrite_moved_includes(args.dry_run)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
