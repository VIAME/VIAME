"""Record what OpenCV's calib3d answers, for `library/measurement/projection`.

`project_point`, `undistort_point`, `stereo_rectify` and
`rectification_maps` are `cv::projectPoints`, `cv::undistortPoints`,
`cv::stereoRectify` and `cv::initUndistortRectifyMap` rewritten without
OpenCV. This is the recording they are held to.

Unlike `tests/golden/math/record_from_eigen.cxx`, the dependency this
records is not going away -- cv2 is a wheel and stays after P7-T09 -- so
this can be re-run at any time:

    source .../setup_viame.sh
    python tests/golden/projection/record_from_opencv.py > opencv.json

The rigs are the ones VIAME actually meets: a near-parallel pair with no
distortion (the golden measurement scene), the camtrawl rig shipped in
`examples/size_measurement`, a strongly distorted wide-angle pair, and a
vertical rig, which is the branch `stereo_rectify` takes when the baseline
runs down the image rather than across it.
"""

import json
import sys

import numpy as np


def rodrigues(vector):
    return cv2.Rodrigues(np.asarray(vector, dtype=np.float64))[0]


def intrinsics(fx, fy, cx, cy, skew=0.0):
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def rigs():
    yield {
        "name": "golden_scene",
        "width": 640, "height": 480,
        "k_left": intrinsics(600.0, 600.0, 319.5, 239.5),
        "d_left": np.zeros(5),
        "k_right": intrinsics(610.0, 610.0, 319.5, 239.5),
        "d_right": np.zeros(5),
        "rotation": rodrigues([0.004, -0.02, 0.001]),
        "translation": np.array([-120.0, 2.0, 5.0]),
    }

    yield {
        "name": "camtrawl",
        "width": 1600, "height": 1200,
        "k_left": intrinsics(1107.73370687, 1101.74223136,
                             822.50433341, 615.35839208),
        "d_left": np.array([-0.08356395, 0.05952725, 0.00101089,
                            -0.00052874, 0.01190306]),
        "k_right": intrinsics(1102.5473414, 1097.27877348,
                              819.33410416, 615.99312957),
        "d_right": np.array([-0.0904797, 0.09477187, 0.00010176,
                             -0.00087904, -0.02854945]),
        "rotation": np.array([[0.99976404, 0.01697303, 0.01355668],
                              [-0.01704124, 0.99984262, 0.00493203],
                              [-0.01347105, -0.00516175, 0.99989591]]),
        "translation": np.array([-209.69904545, 4.07209465, 4.46557305]),
    }

    yield {
        "name": "wide_angle",
        "width": 1280, "height": 720,
        "k_left": intrinsics(420.0, 418.0, 639.5, 359.5),
        "d_left": np.array([-0.31, 0.12, 0.002, -0.001, -0.02]),
        "k_right": intrinsics(425.0, 423.0, 641.0, 358.0),
        "d_right": np.array([-0.29, 0.10, -0.0015, 0.0022, -0.018]),
        "rotation": rodrigues([0.01, 0.09, -0.004]),
        "translation": np.array([-300.0, -6.0, 12.0]),
    }

    yield {
        "name": "vertical",
        "width": 800, "height": 600,
        "k_left": intrinsics(700.0, 700.0, 399.5, 299.5),
        "d_left": np.zeros(5),
        "k_right": intrinsics(700.0, 700.0, 399.5, 299.5),
        "d_right": np.zeros(5),
        "rotation": rodrigues([-0.008, 0.003, 0.002]),
        "translation": np.array([3.0, -150.0, 2.0]),
    }


# Where the sampled points go, in image pixels and in camera millimetres.
IMAGE_POINTS = ((0.0, 0.0), (100.5, 40.25), (319.5, 239.5),
                (511.0, 383.0), (12.75, 470.5), (639.0, 479.0))

CAMERA_POINTS = ((0.0, 0.0, 2000.0), (-350.0, -120.0, 2100.0),
                 (420.0, 260.0, 1500.0), (-90.0, 300.0, 4400.0),
                 (55.0, -240.0, 900.0))

# Where the rectification maps are sampled. A whole map is megabytes; these
# nine positions pin the geometry and the file stays readable.
MAP_SAMPLES = ((0, 0), (1, 1), (7, 11), (100, 80), (317, 239),
               (400, 300), (13, 470), (639, 479), (5, 5))


def numbers(values):
    return [float(value) for value in np.asarray(values).reshape(-1)]


def main():
    out = {"projection": [], "undistortion": [], "rectification": []}

    for rig in rigs():
        size = (rig["width"], rig["height"])
        zero = np.zeros(3)

        for side in ("left", "right"):
            k = rig["k_" + side]
            d = rig["d_" + side]

            projected, _ = cv2.projectPoints(
                np.asarray(CAMERA_POINTS, dtype=np.float64).reshape(-1, 1, 3),
                zero, zero, k, d)

            out["projection"].append({
                "rig": rig["name"], "side": side,
                "intrinsics": numbers(k), "distortion": numbers(d),
                "points": numbers(CAMERA_POINTS),
                "expected": numbers(projected),
            })

            # Through the intrinsic matrix again, which is the identity case
            # every caller but the rectification uses.
            undistorted = cv2.undistortPoints(
                np.asarray(IMAGE_POINTS, dtype=np.float64).reshape(-1, 1, 2),
                k, d, R=np.eye(3), P=k)

            out["undistortion"].append({
                "rig": rig["name"], "side": side,
                "intrinsics": numbers(k), "distortion": numbers(d),
                "points": numbers(IMAGE_POINTS),
                "expected": numbers(undistorted),
            })

        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            rig["k_left"], rig["d_left"], rig["k_right"], rig["d_right"],
            size, rig["rotation"], rig["translation"],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        entry = {
            "rig": rig["name"],
            "width": rig["width"], "height": rig["height"],
            "k_left": numbers(rig["k_left"]),
            "d_left": numbers(rig["d_left"]),
            "k_right": numbers(rig["k_right"]),
            "d_right": numbers(rig["d_right"]),
            "rotation": numbers(rig["rotation"]),
            "translation": numbers(rig["translation"]),
            "r1": numbers(r1), "r2": numbers(r2),
            "p1": numbers(p1), "p2": numbers(p2), "q": numbers(q),
        }

        for side, k, d, r, p in (("left", rig["k_left"], rig["d_left"], r1, p1),
                                 ("right", rig["k_right"], rig["d_right"],
                                  r2, p2)):
            map_x, map_y = cv2.initUndistortRectifyMap(
                k, d, r, p, size, cv2.CV_32FC1)

            # Flat, four numbers per sample: x, y, and where they come
            # from. `tests/library/golden_json.h` reads flat arrays only.
            samples = []
            for x, y in MAP_SAMPLES:
                if x >= rig["width"] or y >= rig["height"]:
                    continue
                samples += [float(x), float(y), float(map_x[y, x]),
                            float(map_y[y, x])]

            entry["map_" + side] = samples

        out["rectification"].append(entry)

    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    import cv2  # noqa: E402  -- after the docstring, so --help needs no cv2
    globals()["cv2"] = cv2
    sys.exit(main())
