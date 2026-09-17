#!/usr/bin/env python3
"""Patch the installed python packages VIAME cannot use unmodified.

One package needs an edit after it is installed.

**`ubelt.ensure_unicode` was removed.** `kwplot` still calls it. It returned
its argument as `str`, so `str(` is what it meant.

There were three, until upstream vendored `torch_liberator` and `liberator`
into `plugins/pytorch/netharn/` with their fixes already applied.

This replaces the block in `cmake/custom_install_viame.cmake` that did the
same edits with `ReplaceStringInFile`, a helper that reads a file, replaces
a string and writes it back -- and says nothing at all when the string is
not there. A patch that silently stops applying is how a dependency bump
turns into a runtime failure three weeks later, so this one reports what it
did and treats "matched nothing, and the result is not already in place" as
an error.

P9 removes this entirely: the patched packages become wheels built by the
wheel CI, and an installed tree stops being something that gets edited.

Usage:
    apply.py --site-packages <dir> [--check]
"""

import argparse
import os
import sys


# ( package, file within it, old, new )
#
# `new` is also how "already applied" is recognised, so each pair has to be
# one the edit makes idempotent -- which they are: none of these `new`
# strings contains its own `old`.
PATCHES = [
    # ubelt.ensure_unicode, removed upstream
    ("kwplot", "mpl_core.py", "ub.ensure_unicode(", "str("),
    ("kwplot", "mpl_multiplot.py", "ub.ensure_unicode(", "str("),
]

# Five patches stood here, four to `torch_liberator` and one to `liberator`:
# three for `torch.load`'s `weights_only` default and two more for
# `ensure_unicode`. Upstream vendored both packages into
# `plugins/pytorch/netharn/` with the fixes already in the source
# (584fe14d6, f540e09f6), so there is nothing installed to patch -- checked
# on the vendored copies, which carry `weights_only=False` and no
# `ub.ensure_unicode`. They came out of `base.in` at the same time.


def apply_one(site_packages, package, relative, old, new, check):
    """(state, detail) for one patch, where state is applied/already/absent/stale."""
    path = os.path.join(site_packages, package, relative)

    if not os.path.isdir(os.path.join(site_packages, package)):
        return "absent", "{} is not installed".format(package)

    if not os.path.isfile(path):
        return "stale", "{} has no {}".format(package, relative)

    with open(path, encoding="utf-8") as handle:
        text = handle.read()

    if old not in text:
        if new in text:
            return "already", path
        return "stale", "{}: neither the old text nor the new is in {}".format(
            package, relative)

    if check:
        return "applied", path

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.replace(old, new))

    return "applied", path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", required=True)
    parser.add_argument("--check", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    counts = {"applied": 0, "already": 0, "absent": 0, "stale": 0}
    stale = []

    for package, relative, old, new in PATCHES:
        state, detail = apply_one(args.site_packages, package, relative,
                                  old, new, args.check)
        counts[state] += 1
        if state == "stale":
            stale.append(detail)

    print("python patches: {} applied, {} already in place, {} for packages "
          "that are not installed".format(
              counts["applied"], counts["already"], counts["absent"]))

    if stale:
        print("\nThese patches matched nothing, and what they would have "
              "produced is not there either. The package has moved and the "
              "patch has to move with it:", file=sys.stderr)
        for detail in stale:
            print("  " + detail, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
