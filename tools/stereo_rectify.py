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

Compute a stereo rectified image pair

"""

import numpy as np
import cv2
import os.path

from optparse import OptionParser


def main():
    usage = "usage: %prog [options] input-image output-image intrinsics extrinsics\n\n"
    usage += "  Rectify a stereo image pair\n"
    parser = OptionParser(usage=usage)

    parser.add_option("-b", "--bayer", default=False,
                      action="store_true", dest="bayer",
                      help="input images are Bayer patterned")

    (options, args) = parser.parse_args()

    if len(args) != 4:
        print(parser.usage)

    img = cv2.imread(args[0])
    left_img = img[:, 0:img.shape[1] // 2]
    right_img = img[:, img.shape[1] // 2:]
    if options.bayer:
        left_img = cv2.cvtColor(left_img[:,:,0], cv2.COLOR_BayerBG2BGR)
        right_img = cv2.cvtColor(right_img[:,:,0], cv2.COLOR_BayerBG2BGR)
    img_shape = left_img.shape[1::-1]

    # read the intrinsics parameters
    fs = cv2.FileStorage(args[2], flags=0)
    M1 = fs.getNode("M1").mat()
    D1 = fs.getNode("D1").mat()
    M2 = fs.getNode("M2").mat()
    D2 = fs.getNode("D2").mat()

    # read the extrinsic parameter
    fs = cv2.FileStorage(args[3], flags=0)
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    Q = fs.getNode("Q").mat()

    map11, map12 = cv2.initUndistortRectifyMap(M1, D1, R1, P1, img_shape, cv2.CV_16SC2)
    map21, map22 = cv2.initUndistortRectifyMap(M2, D2, R2, P2, img_shape, cv2.CV_16SC2)

    left_rect = cv2.remap(left_img, map11, map12, cv2.INTER_CUBIC)
    right_rect = cv2.remap(right_img, map21, map22, cv2.INTER_CUBIC)

    rect_pair = np.hstack((left_rect, right_rect))
    cv2.imwrite(args[1], rect_pair)

if __name__ == "__main__":
    main()

