#!/usr/bin/env python3
"""Rewrite VIAME's C++ namespaces onto `viame` (P11-T01).

    kwiver::vital          -> viame
    kwiver::vital::python  -> viame::python
    kwiver::arrows::<a>    -> viame::<a>
    kwiver::tools          -> viame::tools
    kwiver::<process>      -> viame::<process>
    kwiver::sprokit        -> viame::pipeline
    sprokit                -> viame::pipeline

The longest mapped prefix wins, so `kwiver::vital::streamable` follows
`kwiver::vital` to `viame::streamable` without being listed.

Open decision 8 is answered yes, with the `kwiver` shim lasting one release,
so this rewrites the declarations and the qualified uses and nothing else:

- CMake is left alone. `kwiver::vital_algo` in a `.cmake` file is a target
  name, not a namespace, and those names are the shim: they are defined for
  one more release by `viame-config-targets-install.cmake.in` and by the
  `add_library( kwiver::x ALIAS x )` lines beside each library.
- Header guards keep their names. `KWIVER_VITAL_IMAGE_H` is not an interface.
- Export macros and the target names they are generated from (`vital_algo`,
  `sprokit_pipeline`, ...) are a separate pass: `generate_export_header`
  derives `VITAL_ALGO_EXPORT` from the target, so renaming the macro means
  renaming the target, which means touching the facade list.
- A bare `kwiver` or `sprokit` is only a namespace when it qualifies
  something. Upstream's license URL, a path in a comment and a sentence about
  kwiver all keep the word.
- `library/tpl` is vendored and names none of this.

The namespaces do not map one to one. `kwiver::vital` is two namespaces and
`viame` is one, so an opening pair collapses and one closing brace goes with
it; `sprokit` is one namespace and `viame::pipeline` is two, which the C++17
joined form writes with one brace. Line patterns cannot say which closing
brace belongs to which opening, so this tracks brace depth -- counting only
braces that are code, not the ones in `"{"`, in `'}'`, in a raw string or in
a comment -- and then checks that the file it produced nests the way it
should. A file it cannot follow is named and left alone rather than
half-rewritten.

Both brace styles are namespaces: `namespace kwiver {` and `namespace kwiver`
with the brace on the line below.

Usage:
    rename_namespaces.py [--check] [--verbose] [PATH ...]

With no paths it walks the tracked C++ sources: `.h`, `.cxx`, `.hpp`, `.txx`,
`.cpp` and the `.in` templates of those. `--check` writes nothing.
"""

import argparse
import collections
import os
import re
import subprocess
import sys

CXX_SUFFIXES = (".h", ".cxx", ".hpp", ".txx", ".cpp",
                ".h.in", ".cxx.in", ".hpp.in", ".txx.in", ".cpp.in")

SKIP_PREFIXES = ("packages/", "design/", "library/tpl/")

# Qualified uses, longest first: `kwiver::vital::python` has to be seen
# before `kwiver::vital`.
QUALIFIED = (
    ("kwiver::vital::python", "viame::python"),
    ("kwiver::vital", "viame"),
    ("kwiver::arrows", "viame"),
    ("kwiver::tools", "viame::tools"),
    # The python bindings put their helpers in `kwiver::sprokit::python`,
    # which is sprokit's, not a second `kwiver` sub-namespace.
    ("kwiver::sprokit", "viame::pipeline"),
    ("sprokit", "viame::pipeline"),
    # Last, and not redundant: VIAME's own processes are declared directly in
    # `namespace kwiver` -- `kwiver::refine_detections_process` and the rest of
    # what the `register_processes.cxx` files name. Their declarations follow
    # ("kwiver",) to `viame`, so their uses have to as well.
    ("kwiver", "viame"),
    # Written from inside `namespace kwiver`, where `vital::pluggable` meant
    # `kwiver::vital::pluggable` and `arrows::ocv::x` meant `kwiver::arrows`.
    # Neither namespace exists afterwards, so every one of these is a name
    # that would no longer resolve.
    ("vital", "viame"),
    ("arrows", "viame"),
)

# `using namespace <old>;`, which names a namespace without qualifying anything.
USING = (
    ("kwiver::vital", "viame"),
    ("kwiver::sprokit", "viame::pipeline"),
    ("kwiver", "viame"),
    ("sprokit", "viame::pipeline"),
    ("vital", "viame"),
    ("arrows", "viame"),
)

# A namespace path as declared, to what it becomes.
DECLARED = {
    ("kwiver",): ("viame",),
    ("kwiver", "vital"): ("viame",),
    ("kwiver", "vital", "python"): ("viame", "python"),
    ("kwiver", "arrows"): ("viame",),
    ("kwiver", "tools"): ("viame", "tools"),
    ("kwiver", "sprokit"): ("viame", "pipeline"),
    ("sprokit",): ("viame", "pipeline"),
}

NS_OPEN = re.compile(r"^(\s*)namespace\s+([A-Za-z_][\w:]*)\s*(\{?)\s*(//.*)?$")
NS_CLOSE = re.compile(r"^(\s*)((?:\}\s*)+)(//.*|/\*.*)?$")
BRACE_ONLY = re.compile(r"^\s*\{\s*(//.*)?$")
OLD_NAME = re.compile(r"(?<![\w:])(kwiver|sprokit)(::|\s*$|\s*\{)")

# A qualified name split over a macro's line continuation: `&kwiver::` on one
# line and `vital::` on the next are one name, and rewriting the lines
# separately makes `viame::viame::`. This is rare enough to report rather than
# join -- KWIVER_UNIQUE_PTR in pluggable_macro_magic.h was the only one.
SPLIT_NAME = re.compile(r"(?<![\w:])(kwiver|sprokit|vital|arrows|viame)::\s*\\\s*$")
RAW_START = re.compile(r'R"([^(\s\\]{0,16})\(')

CLEAN = (False, None)      # not in a block comment, not in a raw string


class Unfollowable(Exception):
    """The file's braces do not line up with its namespaces."""


def tracked_sources(paths):
    if paths:
        out = []
        for p in paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    out += [os.path.join(root, f) for f in files
                            if f.endswith(CXX_SUFFIXES)]
            elif p.endswith(CXX_SUFFIXES):
                out.append(p)
        return sorted(out)

    listed = subprocess.run(["git", "ls-files"], capture_output=True,
                            text=True, check=True).stdout.split()
    return sorted(f for f in listed
                  if f.endswith(CXX_SUFFIXES)
                  and not f.startswith(SKIP_PREFIXES))


def split_path(name):
    return tuple(p for p in name.split("::") if p)


def map_path(full):
    """What a declared namespace path becomes, or None to leave it alone."""
    for cut in range(len(full), 0, -1):
        mapped = DECLARED.get(full[:cut])
        if mapped is not None:
            return mapped + full[cut:]
    return None


def code_of(line, state):
    """The part of a line that is code, and the state the next line starts in.

    Braces in `"{"`, in `'}'`, in `R"json({...})json"` and in comments are not
    braces. The state is (inside a block comment, the raw string delimiter we
    are looking for).
    """
    in_comment, raw = state
    out = []
    i = 0
    n = len(line)
    while i < n:
        if raw is not None:
            end = line.find(')' + raw + '"', i)
            if end < 0:
                return "".join(out), (in_comment, raw)
            i = end + len(raw) + 2
            raw = None
            continue
        if in_comment:
            end = line.find("*/", i)
            if end < 0:
                return "".join(out), (True, None)
            i = end + 2
            in_comment = False
            continue
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            in_comment = True
            i += 2
            continue
        m = RAW_START.match(line, i)
        if m:
            raw = m.group(1)
            i = m.end()
            continue
        c = line[i]
        if c in "\"'":
            quote = c
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out), (in_comment, raw)


def rewrite_qualified(line):
    """`kwiver::vital::image` -> `viame::image`, in one line.

    A line that opens a namespace scope is left alone. The opening branch
    handles the ones it understands; what reaches here is a line this cannot
    follow -- `namespace kwiver { namespace vital {`, both on one line -- and
    renaming the first word of it would leave `viame::vital` behind, which is
    worse than leaving the line for a person and saying so.

    An alias is not that: `namespace kv = kwiver::vital;` opens nothing and is
    rewritten like any other use.
    """
    if re.match(r"^\s*namespace\b", line) and "{" in line:
        return line
    # `using namespace kwiver;` has no braces and no `::`, so nothing below
    # would see it.
    for old, new in USING:
        line = re.sub(r"\busing\s+namespace\s+" + re.escape(old) + r"\s*;",
                      "using namespace " + new + ";", line)

    # A leading `::` is part of the name, not something in front of it: the
    # trampolines are full of `::kwiver::vital::timestamp`. It is kept.
    lead = r"(?<![\w:])(::)?"
    for old, new in QUALIFIED:
        if "::" in old:
            line = re.sub(lead + re.escape(old) + r"(?![\w])", r"\1" + new, line)
        else:
            # A bare name only counts when it qualifies something. Without the
            # `::` this rewrites the word wherever it appears: upstream's
            # license URL, a path in a comment, a sentence about kwiver.
            line = re.sub(lead + re.escape(old) + r"::", r"\1" + new + "::", line)
    return line


def opening_at(lines, i, state):
    """The namespace opening at `lines[i]`, or None.

    Returns (path, indent, comment, lines used): two when the brace is on the
    line below, which is the other style in this tree.
    """
    in_comment = state[0] or state[1] is not None
    if in_comment:
        return None
    m = NS_OPEN.match(lines[i])
    if not m:
        return None
    if m.group(3) == "{":
        return split_path(m.group(2)), m.group(1), m.group(4) or "", 1
    if i + 1 < len(lines) and BRACE_ONLY.match(lines[i + 1]):
        return split_path(m.group(2)), m.group(1), m.group(4) or "", 2
    return None


def namespace_paths(text, mapped=False):
    """How a file nests namespaces, in the order they open.

    Brace depth decides which closing brace ends which namespace, as in
    `rewrite`: without it a `}` that ends a function pops a namespace and the
    answer is nonsense. With `mapped`, each path is what it becomes, and a
    namespace that lands on the one already enclosing it is left out -- which
    is what collapsing `kwiver::vital` into `viame` does to it.
    """
    lines = text.split("\n")
    shape = []
    stack = []          # (path as declared, path it becomes)
    depth = []
    braces = 0
    state = CLEAN
    i = 0

    while i < len(lines):
        line = lines[i]
        code, next_state = code_of(line, state)

        opened = opening_at(lines, i, state)
        if opened:
            path, _, _, used = opened
            full = (stack[-1][0] + path) if stack else path
            here = stack[-1][1] if stack else ()
            if mapped:
                target = map_path(full)
                emitted = target if target is not None else (here + path)
            else:
                emitted = full
            stack.append((full, emitted))
            depth.append(braces)
            braces += 1
            if not (mapped and emitted == here):
                shape.append(emitted)
            state = next_state if used == 1 else code_of(lines[i + 1], next_state)[1]
            i += used
            continue

        close = NS_CLOSE.match(line)
        closing = code.count("}") if close else 0
        if (close and closing and stack and len(stack) >= closing
                and braces - closing == depth[-closing]):
            for _ in range(closing):
                stack.pop()
                depth.pop()
                braces -= 1
            state = next_state
            i += 1
            continue

        braces += code.count("{") - code.count("}")
        state = next_state
        i += 1

    return shape


def rewrite(text):
    """Return the rewritten text, namespaces changed and braces dropped."""
    lines = text.split("\n")
    out = []
    stack = []          # (path as declared, path it becomes, what it now says)
    depth = []          # brace depth at each namespace opening
    braces = 0
    changed = 0
    dropped = 0
    skip_blank = False
    state = CLEAN
    i = 0

    while i < len(lines):
        line = lines[i]
        code, next_state = code_of(line, state)

        if skip_blank:
            skip_blank = False
            if not line.strip():
                state = next_state
                i += 1
                continue

        opened = opening_at(lines, i, state)
        if opened:
            path, indent, comment, used = opened
            parent = stack[-1][0] if stack else ()
            full = parent + path
            target = map_path(full)
            here = stack[-1][1] if stack else ()
            brace_line = lines[i + 1] if used == 2 else None

            depth.append(braces)
            braces += 1
            state = next_state if used == 1 else code_of(brace_line, next_state)[1]
            i += used

            # The third element is what this opening now says: None when the
            # line was left alone, so its closing comment is left alone too,
            # and "" when the opening went and its brace goes with it.
            if target is None:
                stack.append((full, (here + path) if here else path, None))
                out.append(line)
                if brace_line is not None:
                    out.append(brace_line)
                continue

            if target == here:
                # Collapsed into the namespace already open: this opening, its
                # closing brace, and a blank line beside each, go.
                stack.append((full, target, ""))
                out.append(None)
                if brace_line is not None:
                    out.append(None)
                changed += 1
                dropped += 1
                skip_blank = True
                continue

            rest = target[len(here):] if target[:len(here)] == here else target
            emitted = "::".join(rest)
            stack.append((full, target, emitted))
            if brace_line is None:
                out.append("{}namespace {} {{{}".format(
                    indent, emitted, (" " + comment.strip()) if comment else ""))
            else:
                out.append("{}namespace {}{}".format(
                    indent, emitted, (" " + comment.strip()) if comment else ""))
                out.append(brace_line)
            changed += 1
            continue

        close = NS_CLOSE.match(line)
        closing = code.count("}") if close else 0
        if (close and closing and stack and len(stack) >= closing
                and braces - closing == depth[-closing]):
            indent = close.group(1)
            comment = (close.group(3) or "").strip()
            pieces = []
            for _ in range(closing):
                full, target, emitted = stack.pop()
                depth.pop()
                braces -= 1
                if emitted == "":
                    continue          # its opening went, so this brace goes
                pieces.append(emitted)
            state = next_state
            i += 1

            if not pieces:
                if out and out[-1] is not None and not out[-1].strip():
                    out[-1] = None
                out.append(None)
                continue

            braces_text = " ".join("}" for _ in pieces)
            if comment and pieces[-1] is not None:
                comment = "// namespace " + pieces[-1]
            out.append("{}{}{}".format(
                indent, braces_text, (" " + comment) if comment else ""))
            continue

        braces += code.count("{") - code.count("}")
        state = next_state
        i += 1
        if braces < 0:
            raise Unfollowable("negative brace depth")
        out.append(rewrite_qualified(line))

    if stack:
        raise Unfollowable("{} namespace(s) left open".format(len(stack)))

    new_text = "\n".join(l for l in out if l is not None)

    want = namespace_paths(text, mapped=True)
    got = namespace_paths(new_text)
    if got != want:
        raise Unfollowable("namespaces nest differently afterwards: "
                           "{} -> {}".format(len(want), len(got)))

    return new_text, changed, dropped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="files or directories")
    parser.add_argument("--check", action="store_true",
                        help="report what would change, write nothing")
    parser.add_argument("--verbose", action="store_true",
                        help="name every file that changes")
    args = parser.parse_args(argv)

    files = tracked_sources(args.paths)
    touched = collections.Counter()
    refused = []
    residue = []
    split = []
    total_ns = 0

    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            refused.append((path, str(exc)))
            continue

        try:
            new_text, changed, dropped = rewrite(text)
        except Unfollowable as exc:
            refused.append((path, str(exc)))
            continue

        if OLD_NAME.search(new_text):
            residue.append(path)

        if SPLIT_NAME.search(text):
            split.append(path)

        if new_text == text:
            continue

        touched[path] = changed
        total_ns += changed
        if args.verbose:
            print("{:<70} {} namespace(s), {} brace(s) dropped".format(
                path, changed, dropped))
        if not args.check:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(new_text)

    print("\n{} of {} files {}, {} namespace declarations rewritten".format(
        len(touched), len(files),
        "would change" if args.check else "changed", total_ns))

    if refused:
        print("\n{} file(s) left alone, to be done by hand:".format(len(refused)))
        for path, why in refused:
            print("  {:<64} {}".format(path, why))

    if residue:
        print("\n{} file(s) still name kwiver or sprokit afterwards:".format(
            len(residue)))
        for path in residue[:20]:
            print("  {}".format(path))

    if split:
        print("\n{} file(s) carry a qualified name across a line continuation, "
              "which is rewritten a line at a time and wants reading:".format(
                  len(split)))
        for path in split:
            print("  {}".format(path))

    return 0


if __name__ == "__main__":
    sys.exit(main())
