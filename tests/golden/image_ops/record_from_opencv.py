#!/usr/bin/env python3
"""Record what OpenCV computes, so `image_ops` can be checked after it is gone.

The counterpart of `tests/golden/math/record_from_eigen.cxx`: run once, by
hand, while OpenCV is still on the path, and commit the result. The C++ test
`tests/library/image_ops/test_color.cxx` reads it and holds the in-house
kernels to it, at the tolerances `design/tasks/phase-07-drop-opencv.md`
states per function.

    source <install>/setup_viame.sh
    python3 tests/golden/image_ops/record_from_opencv.py

It refuses to overwrite an existing recording without --force, for the same
reason the other recorders do: a golden must not be quietly redefined by the
code it is meant to be checking.

The images are committed as flat integer arrays in JSON rather than as PNGs,
because the C++ side has to read them and a JSON reader is thirty lines where
a PNG decoder is the thing under test. They are cropped rather than whole:
the conversions here are per pixel or three by three, so a 32 by 24 window is
as much evidence as a 96 by 64 one and two orders of magnitude less to carry
in the repository. The crop is taken at an even offset so that a Bayer
mosaic keeps its parity.
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import fixtures            # noqa: E402
import opencv_fixtures     # noqa: E402


# The window every case is recorded over: 32 by 24 at (16, 16). Even offsets
# so a Bayer mosaic keeps its parity, and placed over the fixture's disc edge
# and bars so that the window has structure rather than gradient.
CROP_LEFT = 16
CROP_TOP = 16
CROP_WIDTH = 32
CROP_HEIGHT = 24


def crop(array):
    return array[CROP_TOP:CROP_TOP + CROP_HEIGHT,
                 CROP_LEFT:CROP_LEFT + CROP_WIDTH]


def flat(prefix, array):
    """An image as flat `<prefix>_*` keys, so the C++ reader stays trivial.

    Nested objects would need a real parser on the other side; four keys with
    a prefix need none. `data` is in (row, column, plane) order, which is how
    numpy lays it out and how the C++ side walks it.
    """
    if array.ndim == 2:
        array = array[:, :, np.newaxis]

    return {
        prefix + "_width": int(array.shape[1]),
        prefix + "_height": int(array.shape[0]),
        prefix + "_planes": int(array.shape[2]),
        prefix + "_data": [int(v) for v in array.reshape(-1)],
    }


def record():
    import cv2

    rng = np.random.default_rng(fixtures.SEED)
    rgb = fixtures.rgb8(rng)
    gray = fixtures.gray8(rng)
    bayer = opencv_fixtures.build()["bayer_bg"]

    cases = []

    def add(name, source, expected, tolerance, margin=0):
        """One case.

        `tolerance` is the largest per-pixel difference in counts that the
        replacement may show. `margin` is how many pixels of border the
        comparison skips: OpenCV ran on the whole fixture and the recording
        is a window of the result, so a kernel that reads its neighbours sees
        the window's own edge where OpenCV saw real pixels. Only the
        neighbourhood kernels need it, and only by their radius.
        """
        case = {"name": name, "tolerance": tolerance, "margin": margin}
        case.update(flat("input", crop(source)))
        case.update(flat("expected", crop(expected)))
        cases.append(case)

    # Luminance. The same BT.601 fixed point at 14 bits, so the two agree on
    # every pixel but the ones landing exactly on a half -- ten of 768 here,
    # each by one count. OpenCV's own scalar and vector paths disagree there
    # too, so one count is the honest tolerance rather than a concession.
    add("rgb_to_gray", rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1)

    # One plane into three, which is a copy either way
    add("gray_to_rgb", gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 0)

    # Channel order, which is a permutation and therefore exact
    add("swap_rb", rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0)

    # HSV and Lab go through doubles here and through lookup tables in
    # OpenCV, so they agree to a count rather than exactly. Observed maxima
    # when this was recorded: 1 for both. The tolerances are one above, which
    # leaves room for a different OpenCV build without leaving room for a
    # wrong answer.
    add("rgb_to_hsv", rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), 2)
    add("rgb_to_lab", rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab), 2)

    # The round trips, which say the inverses are inverses. Compared against
    # OpenCV's round trip rather than against the original, so that the two
    # implementations lose the same information in the same places.
    # Observed maxima: 1 for both.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    add("hsv_to_rgb", hsv, cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 2)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    add("lab_to_rgb", lab, cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 2)

    # Demosaic. `COLOR_BayerRG2RGB` rather than `BayerBG2RGB`: OpenCV's
    # Bayer constants name the pattern the other way round, so the one that
    # decodes a blue-at-(0,0) mosaic into RGB is the RG one. See color.h.
    # margin 2: the interpolation reads two pixels out, and this is the only
    # case here that reads anything but the pixel under it. Away from the
    # window's edge the two agree exactly -- observed maximum 0 -- so the
    # plan's "demosaic <= 3" is met with room to spare and the tolerance is
    # 1 rather than 3.
    add("demosaic_bg", bayer, cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB), 1,
        margin=2)

    return {
        "recorded": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "opencv": cv2.__version__,
        "note": "Recorded from OpenCV by tests/golden/image_ops/"
                "record_from_opencv.py. Tolerances are per case, in counts.",
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing recording")
    args = parser.parse_args()

    target = os.path.join(HERE, "opencv.json")

    if os.path.exists(target) and not args.force:
        print("{} already recorded; pass --force to re-record".format(target))
        return 1

    payload = record()

    with open(target, "w") as handle:
        json.dump(payload, handle, sort_keys=True, separators=( ",", ":" ))
        handle.write("\n")

    print("recorded {} cases into {}".format(len(payload["cases"]), target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
