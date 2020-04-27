#!/usr/bin/env python

"""ckwg +29
 * Copyright 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

Estimate a disparity image from a rectified image pair

"""

import numpy as np
import cv2
import os.path

from optparse import OptionParser


def disparity(img_left, img_right, disp_range=(0, 240), block_size=11):
    """
    Compute disparity for a fixed disparity range using SGBM

    The image is returned as floating point disparity values
    """
    min_disp = int(disp_range[0])
    num_disp = int(disp_range[1] - disp_range[0])
    # num_disp must be a multiple of 16
    num_disp = ((num_disp + 15) // 16) * 16

    disp_alg = cv2.StereoSGBM_create(numDisparities=num_disp,
                                     minDisparity=min_disp,
                                     uniquenessRatio=10,
                                     blockSize=block_size,
                                     speckleWindowSize=0,
                                     speckleRange=0,
                                     P1=8 * block_size**2,
                                     P2=32 * block_size**2)
    return disp_alg.compute(img_left, img_right).astype('float32') / 16.0


def multipass_disparity(img_left, img_right, outlier_percent=3,
                        range_pad_percent=10):
    """
    Compute dispartity in two passes

    The first pass obtains a robust estimate of the disparity range
    The second pass limits the search to the estimated range for better
    coverage.

    The outlier_percent variable controls which percentange of extreme
    values to ignore when computing the range after the first pass

    The range_pad_percent variable controls by what percentage to expand
    the range by padding on both the low and high ends to account for
    inlier extreme values that were falsely rejected
    """
    # first pass - search the whole range
    disp_img = disparity(img_left, img_right)

    # ignore pixels near the boarder
    border = 20
    disp_img = disp_img[border:-border, border:-border]

    # get a mask of valid disparity pixels
    valid = disp_img >= 0

    # compute a robust range from the valid pixels
    valid_data = disp_img[valid]
    low = np.percentile(valid_data, outlier_percent/2)
    high = np.percentile(valid_data, 100 - outlier_percent/2)
    pad = (high - low) * range_pad_percent / 100.0
    low -= pad
    high += pad
    print("range",low,high)

    # second pass - limit the search range
    disp_img = disparity(img_left, img_right, (low, high))
    valid = disp_img >= low

    disp_img[np.logical_not(valid)] = -1.0
    return disp_img


def scaled_disparity(img_left, img_right):
    img_size = img_left.shape
     # scale the images down by 50%
    img_left = cv2.resize(img_left, (0, 0), fx=0.5, fy=0.5)
    img_right = cv2.resize(img_right, (0, 0), fx=0.5, fy=0.5)

    disp_img = multipass_disparity(img_left, img_right)

    # scale the disparity back up to the original image size
    disp_img = cv2.resize(disp_img, (img_size[1], img_size[0]),
                          interpolation=cv2.INTER_NEAREST)

    # scale the disparity values accordingly
    valid = disp_img >= 0
    disp_img[valid] *= 2.0

    return disp_img


def main():
    usage = "usage: %prog [options] stereo-image\n\n"
    usage += "  Estimate disparity between a pair of rectified images\n"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    if len(args) == 2:
        left_img = cv2.imread(args[0])
        right_img = cv2.imread(args[1])
    elif len(args) == 1:
        img = cv2.imread(args[0])
        left_img = img[:, 0:img.shape[1] // 2]
        right_img = img[:, img.shape[1] // 2:]
    else:
        print("requires two input images or one side-by-side image")

    basename, ext = os.path.splitext(os.path.basename(args[0]))

    # save left and right images
    #cv2.imwrite(basename+"-left.jpg", left_img)
    #cv2.imwrite(basename+"-right.jpg", right_img)

    #left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    #right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    print("computing disparity")
    disp_img = scaled_disparity(left_img, right_img)

    # stretch the range of the disparities to [0,255] for display
    valid = disp_img >= 0
    print("disparity range: ", np.min(disp_img[valid]), np.max(disp_img[valid]))
    disp_img -= np.min(disp_img[valid])
    disp_img *= 255.0 / np.max(disp_img[valid])

    # set the invalid pixels to zero for display
    disp_img[np.logical_not(valid)] = 0

    print("saving "+basename+"-disp.png")
    cv2.imwrite(basename+"-disp.png", disp_img)


if __name__ == "__main__":
    main()

