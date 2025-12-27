#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


"""
Compute a stereo rectified image pair from calibration parameters.
"""

import argparse
import sys

import cv2
import numpy as np


def main():
    """Main entry point for stereo rectification."""
    parser = argparse.ArgumentParser(
        description="Rectify a stereo image pair using calibration parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_image",
                        help="Input side-by-side stereo image (left|right)")
    parser.add_argument("output_image",
                        help="Output rectified stereo image")
    parser.add_argument("intrinsics",
                        help="OpenCV intrinsics file (M1, D1, M2, D2)")
    parser.add_argument("extrinsics",
                        help="OpenCV extrinsics file (R, T, R1, R2, P1, P2, Q)")
    parser.add_argument("-b", "--bayer", action="store_true", default=False,
                        help="Input images are Bayer patterned")

    args = parser.parse_args()

    img = cv2.imread(args.input_image)
    if img is None:
        raise ValueError(f"Failed to read image: {args.input_image}")

    left_img = img[:, 0:img.shape[1] // 2]
    right_img = img[:, img.shape[1] // 2:]

    if args.bayer:
        left_img = cv2.cvtColor(left_img[:, :, 0], cv2.COLOR_BayerBG2BGR)
        right_img = cv2.cvtColor(right_img[:, :, 0], cv2.COLOR_BayerBG2BGR)

    img_shape = left_img.shape[1::-1]

    # Read the intrinsics parameters
    fs = cv2.FileStorage(args.intrinsics, flags=0)
    if not fs.isOpened():
        raise ValueError(f"Failed to open intrinsics file: {args.intrinsics}")
    M1 = fs.getNode("M1").mat()
    D1 = fs.getNode("D1").mat()
    M2 = fs.getNode("M2").mat()
    D2 = fs.getNode("D2").mat()
    fs.release()

    # Read the extrinsic parameters
    fs = cv2.FileStorage(args.extrinsics, flags=0)
    if not fs.isOpened():
        raise ValueError(f"Failed to open extrinsics file: {args.extrinsics}")
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    fs.release()

    # Validate required matrices
    for name, mat in [("M1", M1), ("D1", D1), ("M2", M2), ("D2", D2),
                      ("R1", R1), ("R2", R2), ("P1", P1), ("P2", P2)]:
        if mat is None:
            raise ValueError(f"Matrix {name} not found in calibration files")

    # Compute rectification maps
    map11, map12 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, img_shape, cv2.CV_16SC2)
    map21, map22 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, img_shape, cv2.CV_16SC2)

    # Apply rectification
    left_rect = cv2.remap(left_img, map11, map12, cv2.INTER_CUBIC)
    right_rect = cv2.remap(right_img, map21, map22, cv2.INTER_CUBIC)

    # Save rectified pair
    rect_pair = np.hstack((left_rect, right_rect))
    cv2.imwrite(args.output_image, rect_pair)

    print(f"Saved rectified image to {args.output_image}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
