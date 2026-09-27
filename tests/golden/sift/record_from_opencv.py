#!/usr/bin/env python3
"""Record what OpenCV's SIFT computes, so the port can be checked against it.

Unlike SURF, SIFT is in every opencv-python wheel -- US6711293 expired in 2020
and OpenCV moved it out of `xfeatures2d` -- so this records under the install's
own cv2::

    source build/install/setup_viame.sh
    python3 tests/golden/sift/record_from_opencv.py > tests/golden/sift/opencv.json

**Record under the install's cv2, not the system's.** They are different
versions here (5.0.0 against 4.12.0) and the recordings are the contract.

The tiles come from `tests/golden/surf/`, which already carries them as
lossless PNGs of the grayscale a reader produced, so the colour conversion is
not a variable and there are no new fixtures to commit.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TILES = os.path.join(os.path.dirname(HERE), "surf")

# The defaults `cv2.SIFT_create()` takes, plus one variant per knob. The
# feature count is last because `retainBest` reorders as well as truncates.
CASES = (
    dict(name="default",   features=0,   layers=3, contrast=0.04, edge=10.0, sigma=1.6),
    dict(name="layers4",   features=0,   layers=4, contrast=0.04, edge=10.0, sigma=1.6),
    # Not named "contrast": golden_json finds a key by searching the object's
    # text, so a case *named* the same as one of its own keys is read as that
    # key -- `contrast` came back 0 and the variant silently tested the
    # default with no threshold at all.
    dict(name="contrast08", features=0,  layers=3, contrast=0.08, edge=10.0, sigma=1.6),
    dict(name="edge5",     features=0,   layers=3, contrast=0.04, edge=5.0,  sigma=1.6),
    dict(name="sigma2",    features=0,   layers=3, contrast=0.04, edge=10.0, sigma=2.0),
    dict(name="best200",   features=200, layers=3, contrast=0.04, edge=10.0, sigma=1.6),
)

# Strongest first, and only this many recorded: a 512x512 tile finds a few
# thousand and the whole set is megabytes for nothing the first rows do not
# already show. `count` keeps the true total, so a port that finds a different
# number of keypoints still fails.
KEEP = 250
DESCRIBE = 12


def main():
    import cv2
    import numpy as np

    out = {"opencv": cv2.__version__, "cases": []}

    for tile in ("fish", "seal"):
        path = os.path.join(TILES, tile + ".png")

        if not os.path.exists(path):
            raise SystemExit("fixture missing: %s" % path)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise SystemExit("could not read: %s" % path)

        image = np.ascontiguousarray(image)

        for case in CASES:
            sift = cv2.SIFT_create(case["features"], case["layers"],
                                   case["contrast"], case["edge"],
                                   case["sigma"])
            keypoints, descriptors = sift.detectAndCompute(image, None)

            order = sorted(range(len(keypoints)),
                           key=lambda i: (-keypoints[i].response,
                                          keypoints[i].pt[0],
                                          keypoints[i].pt[1],
                                          keypoints[i].angle))[:KEEP]
            flat = []
            for i in order:
                k = keypoints[i]
                flat += [k.pt[0], k.pt[1], k.size, k.angle, k.response,
                         float(k.octave)]

            entry = dict(case)
            entry.update(
                image=os.path.join("..", "surf", tile + ".png"),
                tile=tile,
                width=int(image.shape[1]), height=int(image.shape[0]),
                count=len(keypoints),
                keypoints=[round(float(v), 6) for v in flat],
                descriptor_width=(0 if descriptors is None
                                  else int(descriptors.shape[1])),
            )

            if descriptors is not None and len(order):
                entry["descriptors"] = [
                    round(float(v), 6)
                    for i in order[:DESCRIBE] for v in descriptors[i]]

            out["cases"].append(entry)

    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
