"""
ckwg +31
Copyright 2016 by Kitware, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Kitware, Inc. nor the names of any contributors may be used
   to endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Test Python interface to vital::camera_intrinsics

"""

import unittest

import nose.tools as ntools
import numpy

from kwiver.vital.types import CameraIntrinsics


class TestVitalCameraIntrinsics (unittest.TestCase):

    def test_default_init(self):
        CameraIntrinsics()

    def test_full_init(self):
        CameraIntrinsics(10, (5, 5), 1.2, 3.1, [4, 5, 6])

    def test_get_focal_length(self):
        ntools.assert_equal(CameraIntrinsics().focal_length, 1.)
        ntools.assert_equal(CameraIntrinsics(5.2).focal_length, 5.2)

    def test_get_principal_point(self):
        numpy.testing.assert_array_equal(
            CameraIntrinsics().principal_point,
            [0, 0]
        )
        numpy.testing.assert_array_equal(
            CameraIntrinsics(principal_point=(10, 2.3)).principal_point,
            [10, 2.3]
        )

    def test_get_aspect_ratio(self):
        ntools.assert_equal(
            CameraIntrinsics().aspect_ratio,
            1.
        )
        ntools.assert_equal(
            CameraIntrinsics(aspect_ratio=2.1).aspect_ratio,
            2.1
        )

    def test_get_skew(self):
        ntools.assert_equal(
            CameraIntrinsics().skew,
            0.
        )
        ntools.assert_equal(
            CameraIntrinsics(skew=1.).skew,
            1.
        )

    def test_get_dist_coeffs(self):
        numpy.testing.assert_array_equal(
            CameraIntrinsics().dist_coeffs,
            numpy.zeros((1,))
        )
        numpy.testing.assert_array_equal(
            CameraIntrinsics(dist_coeffs=(10, 4, 32, 1.1)).dist_coeffs,
            [10, 4, 32, 1.1]
        )

    def test_as_matrix(self):
        numpy.testing.assert_equal(
            CameraIntrinsics().as_matrix(),
            numpy.eye(3)
        )
        numpy.testing.assert_equal(
            CameraIntrinsics(10, (2, 3), 2, 5).as_matrix(),
            [[10, 5, 2],
             [0,  5, 3],
             [0,  0, 1]]
        )

    def test_equal(self):
        ci1 = CameraIntrinsics()
        ci2 = CameraIntrinsics()
        ntools.assert_true(ci1 == ci2)
        ntools.assert_false(ci1 != ci2)

        ci1 = CameraIntrinsics(2, (10, 10), 3, 1)
        ci2 = CameraIntrinsics(2, (11, 10), 3, 0.1)
        ntools.assert_false(ci1 == ci2)
        ntools.assert_true(ci1 != ci2)
