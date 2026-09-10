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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", help="destination under library/")
    parser.add_argument("paths", nargs="+",
                        help="kwiver paths without an extension")
    parser.add_argument("--process", action="store_true",
                        help="these are sprokit processes, not arrow classes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for rel in args.paths:
        moved = move(args.library, rel, args.process, args.dry_run)

        if not moved:
            print("nothing to move for {}".format(rel), file=sys.stderr)
            continue

        pruned = prune_lists(rel, args.dry_run)
        where = deregister(rel, args.process, args.dry_run)

        print("{:52s} -> library/{}  [{}]".format(
            rel, args.library, ", ".join(moved)))

        if not pruned:
            print("   no CMakeLists names it", file=sys.stderr)
        if where is None:
            print("   no registration found", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
