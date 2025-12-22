#!/usr/bin/env python
# ckwg +29
# Copyright 2019-2024 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Estimate a depth image from a rectified image pair and output a PLY point cloud.
"""

import argparse
import os
import sys

import cv2
import numpy as np

from compute_disparity import scaled_disparity
from ply_utilities import write_ply_file


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

    img = cv2.imread(args.stereo_image)
    if img is None:
        raise ValueError(f"Failed to read image: {args.stereo_image}")

    left_img = img[:, 0:img.shape[1] // 2]
    right_img = img[:, img.shape[1] // 2:]

    # Read the matrix for backprojecting to 3D
    fs = cv2.FileStorage(args.extrinsics, flags=0)
    if not fs.isOpened():
        raise ValueError(f"Failed to open extrinsics file: {args.extrinsics}")
    Q = fs.getNode("Q").mat()
    fs.release()

    if Q is None:
        raise ValueError("Q matrix not found in extrinsics file")

    basename, _ = os.path.splitext(os.path.basename(args.stereo_image))

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    print("computing disparity")
    disp_img = scaled_disparity(left_gray, right_gray)

    # Get mask of valid disparity pixels
    valid = disp_img > 0

    img3d = cv2.reprojectImageTo3D(disp_img, Q)

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
