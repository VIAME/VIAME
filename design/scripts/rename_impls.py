#!/usr/bin/env python3
"""Rewrite pipeline files from a replaced implementation's name to its new one.

A replacement keeps the old name registered as an alias, so nothing breaks
either way; this moves the files VIAME ships onto the new names so the aliases
are only carrying add-ons and other people's pipelines.

Add-ons are deliberately left alone: they arrive as zips from
`cmake/download_viame_addons.csv` and are not ours to edit, so they keep
resolving through the aliases. That is the point of keeping the aliases.

Usage:
    rename_impls.py [--check] [PATH ...]

With no paths it rewrites `configs/pipelines`, `configs/gui-params` and
`examples`. `--check` reports what would change without writing.
"""

import argparse
import os
import re
import sys

# Old name -> new name, for the phase 3 VXL replacements.
RENAMES = {
    "vxl_convert_image": "convert_image",
    "vxl_average": "average_frames",
    "vxl_color_commonality": "color_commonality",
    "vxl_morphology": "morphology",
    "vxl_threshold": "threshold",
    "vxl_enhancer": "ocv_enhancer",
    "vxl_homography_guided": "homography_guided",
}

# The bare name `vxl` is an image_io. After the filters above are renamed it
# is the only `vxl` name left registered, so it is renamed wherever it is the
# value of a `:type` key or the implementation segment of a key or block under
# a reader or writer. `.pipe` files write `key   value` and `.conf` files
# write `key = value`, so both separators are accepted.
IMAGE_IO_OLD = "vxl"
IMAGE_IO_NEW = "core"

DEFAULT_PATHS = (
    "configs/pipelines",
    "configs/gui-params",
    "examples",
)

# Never rewritten: not ours to edit.
EXCLUDED = ("configs/add-ons",)

SUFFIXES = (".pipe", ".conf")


def rewrite(text):
    """Return the rewritten text and the number of substitutions made."""
    total = 0

    for old, new in RENAMES.items():
        text, count = re.subn(r"\b{}\b".format(re.escape(old)), new, text)
        total += count

    # `image_reader:type   vxl`, `image_io:type = vxl`, and the like
    text, count = re.subn(
        r"(:type\s*=?\s+){}\b".format(IMAGE_IO_OLD),
        r"\g<1>" + IMAGE_IO_NEW, text)
    total += count

    # `image_reader:vxl:force_byte`, `block image_writer:vxl`, and so on
    text, count = re.subn(
        r"(\b(?:image_reader|image_writer|image_io):){}\b".format(IMAGE_IO_OLD),
        r"\g<1>" + IMAGE_IO_NEW, text)
    total += count

    return text, total


def files(paths):
    for root in paths:
        if not os.path.isdir(root):
            if root.endswith(SUFFIXES):
                yield root
            continue

        for directory, _, names in os.walk(root):
            if any(excluded in directory for excluded in EXCLUDED):
                continue

            for name in sorted(names):
                if name.endswith(SUFFIXES):
                    yield os.path.join(directory, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=None)
    parser.add_argument("--check", action="store_true",
                        help="report without writing")
    args = parser.parse_args()

    changed = 0
    substitutions = 0

    for path in files(args.paths or DEFAULT_PATHS):
        with open(path) as handle:
            text = handle.read()

        rewritten, count = rewrite(text)

        if count == 0:
            continue

        changed += 1
        substitutions += count

        if args.check:
            print("{}: {} substitution(s)".format(path, count))
            continue

        with open(path, "w") as handle:
            handle.write(rewritten)

    print("{} file(s), {} substitution(s){}".format(
        changed, substitutions, " (check only)" if args.check else ""))

    return 0


if __name__ == "__main__":
    sys.exit(main())
