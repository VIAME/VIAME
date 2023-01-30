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

Estimate a depth image from a rectified image pair

"""

from compute_disparity import scaled_disparity
from ply_utilities import write_ply_file

import numpy as np
import cv2
import os.path

from optparse import OptionParser


def main():
    usage = "usage: %prog [options] stereo-image extrinsics\n\n"
    usage += "  Estimate depth between a pair of rectified images\n"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    if len(args) != 2:
        print(parser.usage())

    img = cv2.imread(args[0])
    left_img = img[:, 0:img.shape[1] // 2]
    right_img = img[:, img.shape[1] // 2:]

    # read the matrix for backprojecting to 3D
    fs = cv2.FileStorage(args[1], flags=0)
    Q = fs.getNode("Q").mat()

    basename, ext = os.path.splitext(os.path.basename(args[0]))

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    print("computing disparity")
    disp_img = scaled_disparity(left_gray, right_gray)

    # stretch the range of the disparities to [0,255] for display
    valid = disp_img > 0

    img3d = cv2.reprojectImageTo3D(disp_img, Q)

    pts3d = img3d[valid]
    depths = pts3d[:,2]
    print("depths ", np.min(depths), np.median(depths), np.max(depths))

    bgr = left_img[valid]
    print(bgr.shape)
    color = {}
    color["red"] = bgr[:,2]
    color["green"] = bgr[:,1]
    color["blue"] = bgr[:,0]

    write_ply_file(pts3d, basename+"-points.ply", color, color.keys())

if __name__ == "__main__":
    main()

