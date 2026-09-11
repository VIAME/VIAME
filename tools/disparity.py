#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


"""
Estimate a disparity image from a rectified image pair using SGBM.
"""

import argparse
import os
import sys

import cv2
import numpy as np


def validate_pair(left, right):
    if left is None or right is None or left.shape != right.shape:
        raise ValueError("Left and right images must have the same dimensions and channels")
    if left.ndim not in (2, 3) or min(left.shape[:2]) < 2:
        raise ValueError("Stereo images must be at least 2 x 2 pixels")


def disparity(img_left, img_right, disp_range=(0, 240), block_size=11):
    """Compute disparity for a fixed disparity range using SGBM.

    Args:
        img_left: Left rectified image
        img_right: Right rectified image
        disp_range: Tuple of (min_disparity, max_disparity)
        block_size: Block size for matching

    Returns:
        Disparity image as floating point values
    """
    validate_pair(img_left, img_right)
    if block_size < 1 or block_size % 2 == 0 or block_size > min(img_left.shape[:2]):
        raise ValueError("Block size must be positive, odd, and fit inside the images")
    if not np.isfinite(disp_range).all() or disp_range[1] < disp_range[0]:
        raise ValueError("Disparity range must be finite and ordered")
    min_disp = int(np.floor(disp_range[0]))
    num_disp = max(16, int(np.ceil(disp_range[1] - min_disp)))
    # num_disp must be a multiple of 16
    num_disp = ((num_disp + 15) // 16) * 16

    available = ((img_left.shape[1] - max(0, min_disp) - 1) // 16) * 16
    num_disp = min(num_disp, available)
    if num_disp < 16:
        raise ValueError("Image width is too small for the disparity search")
    channels = 1 if img_left.ndim == 2 else img_left.shape[2]
    disp_alg = cv2.StereoSGBM_create(
        numDisparities=num_disp,
        minDisparity=min_disp,
        uniquenessRatio=10,
        blockSize=block_size,
        speckleWindowSize=0,
        speckleRange=0,
        P1=8 * channels * block_size**2,
        P2=32 * channels * block_size**2
    )
    return disp_alg.compute(img_left, img_right).astype('float32') / 16.0


def multipass_disparity(img_left, img_right, outlier_percent=3,
                        range_pad_percent=10):
    """Compute disparity in two passes.

    The first pass obtains a robust estimate of the disparity range.
    The second pass limits the search to the estimated range for better
    coverage.

    Args:
        img_left: Left rectified image
        img_right: Right rectified image
        outlier_percent: Percentage of extreme values to ignore when
            computing the range after the first pass
        range_pad_percent: Percentage to expand the range by padding on
            both the low and high ends

    Returns:
        Disparity image with invalid pixels set to -1.0
    """
    # First pass - search the whole range
    disp_img = disparity(img_left, img_right)

    # Ignore pixels near the border
    border = min(20, (min(disp_img.shape[:2]) - 1) // 4)
    if border:
        disp_img = disp_img[border:-border, border:-border]

    # Get a mask of valid disparity pixels
    valid = disp_img >= 0

    # Compute a robust range from the valid pixels
    valid_data = disp_img[valid]
    if not valid_data.size:
        return np.full(img_left.shape[:2], -1.0, dtype=np.float32)
    low = np.percentile(valid_data, outlier_percent / 2)
    high = np.percentile(valid_data, 100 - outlier_percent / 2)
    pad = (high - low) * range_pad_percent / 100.0
    low -= pad
    high += pad
    print(f"range {low} {high}")

    # Second pass - limit the search range
    disp_img = disparity(img_left, img_right, (low, high))
    valid = disp_img >= np.floor(low)

    disp_img[np.logical_not(valid)] = -1.0
    return disp_img


def scaled_disparity(img_left, img_right):
    """Compute disparity at half resolution and scale back up.

    Args:
        img_left: Left rectified image
        img_right: Right rectified image

    Returns:
        Disparity image at original resolution
    """
    validate_pair(img_left, img_right)
    img_size = img_left.shape
    # Scale the images down by 50%
    img_left = cv2.resize(img_left, (0, 0), fx=0.5, fy=0.5)
    img_right = cv2.resize(img_right, (0, 0), fx=0.5, fy=0.5)

    disp_img = multipass_disparity(img_left, img_right)

    # Scale the disparity back up to the original image size
    disp_img = cv2.resize(disp_img, (img_size[1], img_size[0]),
                          interpolation=cv2.INTER_NEAREST)

    # Scale the disparity values accordingly
    valid = disp_img >= 0
    disp_img[valid] *= 2.0

    return disp_img


def main():
    """Main entry point for disparity estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate disparity between a pair of rectified images",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs='+',
                        help="One side-by-side image or two separate left/right images")

    parser.add_argument("--numeric-output", help="Write unnormalized float disparities as a NumPy .npy file")
    args = parser.parse_args()

    if len(args.images) == 2:
        left_img = cv2.imread(args.images[0])
        right_img = cv2.imread(args.images[1])
        if left_img is None:
            raise ValueError(f"Failed to read left image: {args.images[0]}")
        if right_img is None:
            raise ValueError(f"Failed to read right image: {args.images[1]}")
    elif len(args.images) == 1:
        img = cv2.imread(args.images[0])
        if img is None:
            raise ValueError(f"Failed to read image: {args.images[0]}")
        left_img = img[:, 0:img.shape[1] // 2]
        right_img = img[:, img.shape[1] // 2:]
    else:
        parser.error("Requires one or two input images")

    basename, _ = os.path.splitext(os.path.basename(args.images[0]))

    print("computing disparity")
    disp_img = scaled_disparity(left_img, right_img)

    # Stretch the range of the disparities to [0,255] for display
    valid = disp_img >= 0
    numeric = disp_img.copy()
    shown = np.zeros(disp_img.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.min(disp_img[valid]), np.max(disp_img[valid])
        print(f"disparity range: {low} {high}")
        shown[valid] = ((disp_img[valid] - low) * 255 / (high - low)).astype(np.uint8) if high > low else 255
    else:
        print("No valid disparities found", file=sys.stderr)
    disp_img = shown
    if args.numeric_output:
        np.save(args.numeric_output, numeric)

    output_file = f"{basename}-disp.png"
    print(f"saving {output_file}")
    if not cv2.imwrite(output_file, disp_img):
        raise ValueError(f"Failed to write disparity image: {output_file}")

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
