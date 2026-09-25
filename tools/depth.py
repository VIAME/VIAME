#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


"""
Estimate a depth image from a rectified image pair and output a PLY point cloud.
"""

import argparse
import os
import sys

import numpy as np

from viame import image_kernels
from viame.measurement import projection
from viame.utilities import imageops, opencv_yaml

from disparity import scaled_disparity
from viame.utilities.utilities_ply import write_ply_file


def main():
    """Main entry point for depth estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate depth from a pair of rectified images",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stereo_image",
                        help="Side-by-side stereo image (left|right)")
    parser.add_argument("extrinsics",
                        help="OpenCV extrinsics file containing Q matrix")

    args = parser.parse_args()

    try:
        img = imageops.read_image(args.stereo_image)
    except OSError as exc:
        raise ValueError(f"Failed to read image: {args.stereo_image}") from exc

    left_img = img[:, 0:img.shape[1] // 2]
    right_img = img[:, img.shape[1] // 2:]

    # Read the matrix for backprojecting to 3D
    try:
        Q = opencv_yaml.read(args.extrinsics, ["Q"])["Q"]
    except OSError as exc:
        raise ValueError(
            f"Failed to open extrinsics file: {args.extrinsics}") from exc

    if Q is None:
        raise ValueError("Q matrix not found in extrinsics file")

    basename, _ = os.path.splitext(os.path.basename(args.stereo_image))

    left_gray = image_kernels.to_gray(np.ascontiguousarray(left_img))
    right_gray = image_kernels.to_gray(np.ascontiguousarray(right_img))

    print("computing disparity")
    disp_img = scaled_disparity(left_gray, right_gray)

    # Get mask of valid disparity pixels
    valid = disp_img > 0

    img3d = projection.reproject_to_3d(disp_img, Q)

    pts3d = img3d[valid]
    depths = pts3d[:, 2]
    print(f"depths: min={np.min(depths):.2f} median={np.median(depths):.2f} max={np.max(depths):.2f}")

    bgr = left_img[valid]
    color = {
        "red": bgr[:, 2],
        "green": bgr[:, 1],
        "blue": bgr[:, 0]
    }

    output_file = f"{basename}-points.ply"
    print(f"saving {output_file}")
    write_ply_file(pts3d, output_file, color, color.keys())

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
