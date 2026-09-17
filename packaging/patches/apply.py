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
# kwimage's putText fails on OpenCV 5, which accepts only uint8 destination
# images: the trainer's evaluation draws annotation text on a float image and
# dies with `(-215) img.depth() == CV_8U` (lite-findings 2.25).
#
# Upstream kwimage ebaba74, "Fix OpenCV 5 drawing and vector compatibility",
# carried verbatim: that commit is on `main` and unreleased, and both the
# newest tag and the newest PyPI release are 0.11.6, which the lock pins.
# Drop this when 0.12.0 ships.
#
# Two substitutions. The first puts the helper at module scope, where
# upstream has it; doing it on the call line instead lands a `def` inside
# `draw_text_on_image`, truncating the function so it returns None and loses
# its `return_info` branch -- which nothing in the suite would notice.
#
# The first substitution also reflows the `_text_sizes` signature, so that
# its own anchor is gone afterwards. `apply_one` reports a patch as already
# applied only when the old text is absent, so an anchor that survives the
# replacement re-inserts the helper on every build.
_KWIMAGE_INSERT_OLD = 'def _text_sizes(text, org, border_thickness, kwargs, valign, halign):'

_KWIMAGE_INSERT_NEW = 'def _cv2_put_text_compat(img, text, xy, kwargs):\n    """Call ``cv2.putText`` across OpenCV 3.x through 5.x.\n\n    OpenCV 5\'s replacement text renderer only accepts uint8 destination\n    images.  Older OpenCV versions accepted other image dtypes, but silently\n    disabled antialiasing for them.  Preserve that behavior by first trying\n    the native call and, only when it rejects a non-uint8 image, rasterizing a\n    binary uint8 text mask and assigning the requested color into the original\n    image.\n\n    The mask fallback avoids quantizing the input image and preserves masked\n    array masks, NaNs outside the text pixels, the input dtype, and inplace\n    behavior.\n    """\n    import cv2\n\n    try:\n        return cv2.putText(img, text, xy, **kwargs)\n    except cv2.error as ex:\n        image_data = np.asarray(img)\n        is_uint8_depth_error = \'img.depth() == CV_8U\' in str(ex)\n        if image_data.dtype == np.uint8 or not is_uint8_depth_error:\n            raise\n\n    mask = np.zeros(image_data.shape[0:2], dtype=np.uint8)\n    mask_kwargs = kwargs.copy()\n    mask_kwargs[\'color\'] = 255\n    if mask_kwargs.get(\'lineType\', None) == cv2.LINE_AA:\n        # OpenCV 4 used non-antialiased drawing for non-uint8 destinations.\n        # Keeping a binary mask also avoids introducing extra intermediate\n        # values into float images that historically only contained the\n        # background and requested drawing colors.\n        mask_kwargs[\'lineType\'] = cv2.LINE_8\n    cv2.putText(mask, text, xy, **mask_kwargs)\n\n    text_pixels = mask != 0\n    color = np.asarray(kwargs[\'color\'])\n    if image_data.ndim == 2:\n        draw_value = color.ravel()[0]\n    else:\n        num_channels = image_data.shape[2]\n        color = color.ravel()\n        if color.size < num_channels:\n            color = np.pad(color, (0, num_channels - color.size))\n        draw_value = color[:num_channels]\n\n    # Write through np.asarray so assigning into a MaskedArray does not alter\n    # its mask.  This also overwrites NaNs where glyph pixels are opaque, as\n    # the old OpenCV implementation did.\n    image_data[text_pixels] = draw_value\n    return img\n\n\ndef _text_sizes(\n        text, org, border_thickness, kwargs, valign, halign):'

_KWIMAGE_CALL_OLD = '        img = cv2.putText(img, line, xy, **kwargs)'

_KWIMAGE_CALL_NEW = '        img = _cv2_put_text_compat(img, line, xy, kwargs)'

PATCHES = [
    # ubelt.ensure_unicode, removed upstream
    ("kwplot", "mpl_core.py", "ub.ensure_unicode(", "str("),
    ("kwplot", "mpl_multiplot.py", "ub.ensure_unicode(", "str("),
    # OpenCV 5 putText, upstream kwimage ebaba74
    ("kwimage", "im_draw.py", _KWIMAGE_INSERT_OLD, _KWIMAGE_INSERT_NEW),
    ("kwimage", "im_draw.py", _KWIMAGE_CALL_OLD, _KWIMAGE_CALL_NEW),
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
