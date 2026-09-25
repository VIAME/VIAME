#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


"""
Compute a stereo rectified image pair from calibration parameters.
"""

import argparse
import sys

import numpy as np

from viame import image_kernels
from viame.measurement import projection
from viame.utilities import imageops, opencv_yaml


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

    # Unchanged, because a Bayer frame is a single plane and a 16-bit one
    # must not be narrowed before it is rectified.
    try:
        img = imageops.read_unchanged(args.input_image)
    except OSError as error:
        raise ValueError(f"Failed to read image: {args.input_image}") from error

    if img.shape[1] % 2:
        raise ValueError("Stitched stereo image width must be even")

    left_img = img[:, 0:img.shape[1] // 2]
    right_img = img[:, img.shape[1] // 2:]

    if args.bayer:
        # "RG" names the mosaic, which is what `cv2.COLOR_BayerBG2BGR` asks
        # for: OpenCV spells its constants by the second row's pair, the
        # reverse of a datasheet. RGB out, where OpenCV gave BGR.
        def debayer(plane):
            plane = plane if plane.ndim == 2 else plane[:, :, 0]
            return image_kernels.demosaic(np.ascontiguousarray(plane), "RG")

        left_img = debayer(left_img)
        right_img = debayer(right_img)

    height, width = left_img.shape[:2]

    # Read the intrinsics parameters
    try:
        found = opencv_yaml.read(args.intrinsics, ("M1", "D1", "M2", "D2"))
    except OSError as error:
        raise ValueError(
            f"Failed to open intrinsics file: {args.intrinsics}") from error
    M1, D1, M2, D2 = (found[k] for k in ("M1", "D1", "M2", "D2"))

    # Read the extrinsic parameters
    try:
        found = opencv_yaml.read(args.extrinsics, ("R1", "R2", "P1", "P2"))
    except OSError as error:
        raise ValueError(
            f"Failed to open extrinsics file: {args.extrinsics}") from error
    R1, R2, P1, P2 = (found[k] for k in ("R1", "R2", "P1", "P2"))

    # Validate required matrices
    for name, mat in [("M1", M1), ("D1", D1), ("M2", M2), ("D2", D2),
                      ("R1", R1), ("R2", R2), ("P1", P1), ("P2", P2)]:
        if mat is None:
            raise ValueError(f"Matrix {name} not found in calibration files")

    for name, mat, shape in [("M1", M1, (3, 3)), ("M2", M2, (3, 3)),
                              ("R1", R1, (3, 3)), ("R2", R2, (3, 3)),
                              ("P1", P1, (3, 4)), ("P2", P2, (3, 4))]:
        if mat.shape != shape or not np.isfinite(mat).all():
            raise ValueError(f"{name} must be a finite {shape} matrix")
    for name, mat in [("D1", D1), ("D2", D2)]:
        if mat.size not in (4, 5, 8, 12, 14) or not np.isfinite(mat).all():
            raise ValueError(f"Invalid distortion coefficients: {name}")

    # Compute rectification maps. Float maps rather than the fixed point
    # `CV_16SC2` pair OpenCV was asked for, which quantises the sample
    # positions to a thirty-second of a pixel.
    map11, map12 = projection.rectification_maps(M1, D1, R1, P1, width, height)
    map21, map22 = projection.rectification_maps(M2, D2, R2, P2, width, height)

    # Apply rectification
    left_rect = image_kernels.remap(np.ascontiguousarray(left_img),
                                    map11, map12, "bicubic")
    right_rect = image_kernels.remap(np.ascontiguousarray(right_img),
                                     map21, map22, "bicubic")

    # Save rectified pair
    rect_pair = np.hstack((left_rect, right_rect))
    try:
        imageops.write_image(args.output_image, rect_pair)
    except OSError as error:
        raise ValueError(
            f"Failed to write rectified image: {args.output_image}") from error

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
