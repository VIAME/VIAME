#!/usr/bin/env python3
"""Record what OpenCV's SURF computes, so a port can be checked against it.

Run under a cv2 built with the non-free modules -- this branch's wheel is not
one, so this is recorded from a build that is (VIAME `main` carries one
through fletch)::

    source <a non-free install>/setup_viame.sh
    python3 tests/golden/surf/record_from_opencv.py > opencv.json

`ocv_SURF` is `cv::xfeatures2d::SURF`, which every opencv-python wheel
excludes because the algorithm is patented. Seven shipped configs select it,
so this branch implements it rather than changing them, and this is the
reference that implementation is held to.
"""
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))            # tests/golden/surf
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
IMAGES = os.path.join(ROOT, "tests", "pipelines", "pipelines_test_data", "images")

# The settings the shipped configs use, plus the defaults, plus the two flags
# that change the descriptor and the orientation pass.
CASES = (
    dict(name="shipped",   hessian=400.0, octaves=4, layers=3, extended=False, upright=False),
    dict(name="default",   hessian=100.0, octaves=4, layers=3, extended=False, upright=False),
    dict(name="upright",   hessian=400.0, octaves=4, layers=3, extended=False, upright=True),
    dict(name="extended",  hessian=400.0, octaves=4, layers=3, extended=True,  upright=False),
    dict(name="octaves2",  hessian=400.0, octaves=2, layers=3, extended=False, upright=False),
)

# Tiles, not whole frames. `seal_1` is 5760x3840 and finds 65k keypoints at
# the shipped threshold, which is minutes of comparison and megabytes of
# reference for nothing the algorithm does not already show on a tile.
#
# Each tile is saved beside this file as a lossless PNG and *that* is what the
# test reads. Recording the grayscale input rather than naming the JPEG takes
# the colour conversion out of the comparison: `IMREAD_GRAYSCALE` and whatever
# the reader on the other side does no longer have to agree.
INPUTS = (
    ("fish", "fish/fish_1_seq_01.jpg", (650, 350, 512, 512)),
    ("seal", "seal_1.jpg", (2600, 1700, 512, 512)),
)

# How many keypoints to record per case, strongest first.
KEEP = 200


def numpy_ascontiguous(a):
    import numpy as np
    return np.ascontiguousarray(a)


def main():
    import cv2
    import numpy as np

    out = {"opencv": cv2.__version__, "cases": []}

    for tile, relative, (x, y, w, h) in INPUTS:
        path = os.path.join(IMAGES, relative)
        if not os.path.exists(path):
            raise SystemExit(f"fixture missing: {path}")
        whole = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if whole is None:
            raise SystemExit(f"could not read: {path}")
        if y + h > whole.shape[0] or x + w > whole.shape[1]:
            raise SystemExit(f"tile {x},{y},{w},{h} does not fit {relative}")
        image = numpy_ascontiguous(whole[y:y + h, x:x + w])

        written = os.path.join(HERE, tile + ".png")
        if not cv2.imwrite(written, image):
            raise SystemExit(f"could not write {written}")

        for case in CASES:
            surf = cv2.xfeatures2d.SURF_create(
                case["hessian"], case["octaves"], case["layers"],
                case["extended"], case["upright"])
            keypoints, descriptors = surf.detectAndCompute(image, None)

            # Strongest first, so a truncated comparison is still meaningful,
            # and flat, because tests/golden/golden_json.h reads flat arrays.
            # Strongest KEEP only: `seal_1` alone finds 65k at the shipped
            # threshold, and recording them all makes a 35 MB golden file. The
            # true total is kept in `count`, so a port that finds a different
            # number still fails.
            order = sorted(range(len(keypoints)),
                           key=lambda i: -keypoints[i].response)[:KEEP]
            flat = []
            for i in order:
                k = keypoints[i]
                flat += [k.pt[0], k.pt[1], k.size, k.angle, k.response,
                         float(k.octave), float(k.class_id)]

            entry = dict(case)
            # 0/1, not true/false: the reader on the other side parses every
            # value with strtod, which turns `true` into 0 and would test the
            # plain descriptor twice while reporting it had tested both.
            entry["extended"] = int(case["extended"])
            entry["upright"] = int(case["upright"])
            entry.update(
                image=tile + ".png",
                source=relative,
                width=int(image.shape[1]), height=int(image.shape[0]),
                count=len(keypoints),
                keypoints=[round(v, 6) for v in flat],
                descriptor_width=0 if descriptors is None else int(descriptors.shape[1]),
            )
            # Descriptors for the strongest few only: the whole set is
            # megabytes and the first rows are what a mismatch shows up in.
            if descriptors is not None and len(order):
                keep = order[:8]
                entry["descriptors"] = [
                    round(float(v), 6) for i in keep for v in descriptors[i]]
            out["cases"].append(entry)

    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
