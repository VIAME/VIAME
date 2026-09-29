#!/usr/bin/env python3
"""Regenerate the python-build-standalone pin table in the CMake module.

The table in `cmake/viame_python_standalone.cmake` is a patch version per
CPython series and a SHA256 per platform -- 25 hashes. Copying those by hand
is how a pin ends up pointing at an archive nobody checked, so this reads
them from the release's own SHA256SUMS and rewrites the block in place.

    python3 cmake/wheel/update_python_pins.py --release 20260901

Without `--release` it re-reads the release the module already names, which
is how to verify that what is committed matches the published sums: the
script exits non-zero and prints a diff if the file would change, so it can
run in CI.
"""
import argparse
import io
import re
import sys
import urllib.request
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "viame_python_standalone.cmake"

SUMS_URL = ("https://github.com/astral-sh/python-build-standalone"
            "/releases/download/{release}/SHA256SUMS")

# The series VIAME builds wheels for, and the platforms it builds them on.
# A series is added here and in the `STRINGS` property beside the option.
SERIES = ["3.10", "3.11", "3.12", "3.13", "3.14"]

TRIPLES = [
    ("x86_64-pc-windows-msvc",     "windows x86_64"),
    ("aarch64-apple-darwin",       "macos arm64"),
    ("x86_64-apple-darwin",        "macos x86_64"),
    ("aarch64-unknown-linux-gnu",  "linux aarch64"),
    ("x86_64-unknown-linux-gnu",   "linux x86_64"),
]

# `install_only` archives unpack to a `python/` that runs from wherever it is
# put, which is the whole reason this distribution is the one used.
ENTRY = re.compile(
    r"^([0-9a-f]{64})\s+"
    r"cpython-(\d+\.\d+\.\d+)\+(\d+)-([a-z0-9_.\-]+)-install_only\.tar\.gz$")

BEGIN = "# BEGIN generated pins -- cmake/wheel/update_python_pins.py"
END = "# END generated pins"


def read_sums(release):
    url = SUMS_URL.format(release=release)
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8")


def collect(text, release):
    """{series: (patch_version, {triple: sha})} for the series we build."""
    wanted = {t for t, _ in TRIPLES}
    found = {}

    for line in text.splitlines():
        match = ENTRY.match(line.strip())
        if not match:
            continue
        sha, version, archive_release, triple = match.groups()
        if archive_release != release or triple not in wanted:
            continue
        series = ".".join(version.split(".")[:2])
        if series not in SERIES:
            continue
        # A release publishes one patch per series; assert rather than
        # silently taking the last, which would make the pin order-dependent.
        patch, shas = found.setdefault(series, (version, {}))
        if patch != version:
            sys.exit("two patch versions for %s in %s: %s and %s"
                     % (series, release, patch, version))
        shas[triple] = sha

    missing = [s for s in SERIES if s not in found]
    if missing:
        sys.exit("release %s publishes nothing for %s"
                 % (release, ", ".join(missing)))

    for series, (_, shas) in found.items():
        gaps = [t for t, _ in TRIPLES if t not in shas]
        if gaps:
            sys.exit("release %s has no %s archive for %s"
                     % (release, ", ".join(gaps), series))

    return found


def render(found):
    lines = [BEGIN]
    for series in SERIES:
        version, shas = found[series]
        key = series.replace(".", "_")
        lines.append("")
        lines.append('set( _standalone_patch_%s "%s" )' % (key, version))
        for triple, label in TRIPLES:
            lines.append("set( _standalone_sha_%s_%s"
                         % (key, triple.replace("-", "_")))
            lines.append('     "%s" )  # %s' % (shas[triple], label))
    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",
                        help="python-build-standalone release tag "
                             "(default: the one the module names)")
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the file would change")
    args = parser.parse_args()

    source = io.open(MODULE, encoding="utf-8").read()

    named = re.search(r"set\( _standalone_release (\d+) \)", source)
    if not named:
        sys.exit("no _standalone_release in %s" % MODULE)

    release = args.release or named.group(1)

    if BEGIN not in source or END not in source:
        sys.exit("no generated-pin markers in %s; add %r and %r around the "
                 "table before running this" % (MODULE, BEGIN, END))

    found = collect(read_sums(release), release)

    updated = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END),
                     lambda _: render(found), source, flags=re.S)
    updated = updated.replace("set( _standalone_release %s )" % named.group(1),
                              "set( _standalone_release %s )" % release, 1)

    if updated == source:
        print("pins are up to date for release %s" % release)
        return 0

    if args.check:
        print("pins differ from release %s" % release, file=sys.stderr)
        return 1

    io.open(MODULE, "w", encoding="utf-8", newline="").write(updated)
    print("rewrote the pin table for release %s" % release)
    for series in SERIES:
        print("  %-5s %s" % (series, found[series][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
