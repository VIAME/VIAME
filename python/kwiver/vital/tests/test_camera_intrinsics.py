"""
ckwg +31
Copyright 2016-2020 by Kitware, Inc.
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

import nose.tools as nt
import numpy as np

from kwiver.vital.types import CameraIntrinsics, SimpleCameraIntrinsics

# TODO:
# Test overriding methods with trampoline
# Tests for base camera_intrinsics class
# Tests for rest of methods in simple_ci
# Tests for inherited methods in simple_ci
# TODO below
# figure out whats going on with max_distort_radius and others

class TestVitalSimpleCameraIntrinsics(unittest.TestCase):
    def setUp(self):
        self.focal_length = 10.5
        self.principal_point = [3.14, 6.28]
        self.aspect_ratio = 1.2
        self.skew = 3.1
        self.dist_coeffs = [4.5, 5.2, 6.8]
        self.image_width = 1080
        self.image_height = 720

        # Calibration matrix constructed so that resulting
        # camera_intrinsics properties should be equal to one constructed using above
        # parameters directly. Also matches format of .matrix()
        self.K = np.array(
            [
                [self.focal_length, self.skew, self.principal_point[0]],
                [0, self.focal_length / self.aspect_ratio, self.principal_point[1]],
                [0, 0, 1],
            ]
        )

    def check_cam_intrins_properties_equal(
        self,
        ci,
        focal_length=None,
        principal_point=None,
        aspect_ratio=None,
        skew=None,
        dist_coeffs=None,
        image_width=None,
        image_height=None,
        max_distort_radius_sq=float("inf"),
        check_width_height=True,
    ):
        focal_length = self.focal_length if focal_length is None else focal_length
        principal_point = (
            self.principal_point if principal_point is None else principal_point
        )
        aspect_ratio = self.aspect_ratio if aspect_ratio is None else aspect_ratio
        skew = self.skew if skew is None else skew
        dist_coeffs = self.dist_coeffs if dist_coeffs is None else dist_coeffs
        image_width = self.image_width if image_width is None else image_width
        image_height = self.image_height if image_height is None else image_height

        nt.assert_almost_equal(ci.focal_length(), focal_length)
        np.testing.assert_array_almost_equal(ci.principal_point(), principal_point)
        nt.assert_almost_equal(ci.aspect_ratio(), aspect_ratio)
        nt.assert_almost_equal(ci.skew(), skew)
        np.testing.assert_array_almost_equal(ci.dist_coeffs(), dist_coeffs)

        if check_width_height:
            nt.assert_almost_equal(ci.image_width(), image_width)
            nt.assert_almost_equal(ci.image_height(), image_height)

        nt.assert_equal(ci.get_max_distort_radius_sq(), max_distort_radius_sq)
        nt.assert_equal(ci.max_distort_radius(), np.sqrt(max_distort_radius_sq))

    def test_default_init(self):
        SimpleCameraIntrinsics()

    def test_full_init(self):
        s = SimpleCameraIntrinsics(
            self.focal_length,
            self.principal_point,
            self.aspect_ratio,
            self.skew,
            self.dist_coeffs,
            self.image_width,
            self.image_height,
        )
        self.check_cam_intrins_properties_equal(s)

    def test_full_init_defaults(self):
        s = SimpleCameraIntrinsics(self.focal_length, self.principal_point)
        self.check_cam_intrins_properties_equal(
            s,
            aspect_ratio=1.0,
            skew=0.0,
            dist_coeffs=np.array([]),
            image_width=0,
            image_height=0,
        )

    def test_full_init_kwargs(self):
        s = SimpleCameraIntrinsics(
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            aspect_ratio=self.aspect_ratio,
            skew=self.skew,
            dist_coeffs=self.dist_coeffs,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        self.check_cam_intrins_properties_equal(s)

    def test_init_from_base(self):
        # TODO: use py_camera_intrinsics here
        pass

    # TODO
    def test_init_from_string(self):
        pass

    def test_init_from_calibration_mat(self):
        s = SimpleCameraIntrinsics(self.K, self.dist_coeffs)
        # This constructor doesn't initialize the width and height of the image,
        # so we wont check those
        self.check_cam_intrins_properties_equal(s, check_width_height=False)

    def test_init_from_calibration_mat_defaults(self):
        s = SimpleCameraIntrinsics(self.K)
        self.check_cam_intrins_properties_equal(
            s, dist_coeffs=np.array([]), check_width_height=False
        )

    def test_get_set_focal_length(self):
        s = SimpleCameraIntrinsics()
        s.set_focal_length(self.focal_length)
        nt.assert_almost_equal(s.focal_length(), self.focal_length)

    def test_get_set_principal_point(self):
        s = SimpleCameraIntrinsics()
        s.set_principal_point(self.principal_point)
        np.testing.assert_array_almost_equal(s.principal_point(), self.principal_point)

    def test_get_set_aspect_ratio(self):
        s = SimpleCameraIntrinsics()
        s.set_aspect_ratio(self.aspect_ratio)
        nt.assert_almost_equal(s.aspect_ratio(), self.aspect_ratio)

    def test_get_set_skew(self):
        s = SimpleCameraIntrinsics()
        s.set_skew(self.skew)
        nt.assert_almost_equal(s.skew(), self.skew)

    def test_get_set_image_width(self):
        s = SimpleCameraIntrinsics()
        s.set_image_width(self.image_width)
        nt.assert_almost_equal(s.image_width(), self.image_width)

    def test_get_set_image_height(self):
        s = SimpleCameraIntrinsics()
        s.set_image_height(self.image_height)
        nt.assert_almost_equal(s.image_height(), self.image_height)

    def test_get_set_dist_coeffs(self):
        s = SimpleCameraIntrinsics()
        s.set_dist_coeffs(self.dist_coeffs)
        np.testing.assert_array_almost_equal(s.dist_coeffs(), self.dist_coeffs)

    # TODO
    def test_max_distort_radius(self):
        pass

    # TODO
    def test_max_distort_radius_sq(self):
        pass

    # TODO
    def test_str(self):
        pass

    # def test_as_matrix(self):
    #     numpy.testing.assert_equal(CameraIntrinsics().as_matrix(), numpy.eye(3))
    #     numpy.testing.assert_equal(
    #         CameraIntrinsics(10, (2, 3), 2, 5).as_matrix(),
    #         [[10, 5, 2], [0, 5, 3], [0, 0, 1]],
    #     )

    # def test_equal(self):
    #     ci1 = CameraIntrinsics()
    #     ci2 = CameraIntrinsics()
    #     ntools.assert_true(ci1 == ci2)
    #     ntools.assert_false(ci1 != ci2)

    #     ci1 = CameraIntrinsics(2, (10, 10), 3, 1)
    #     ci2 = CameraIntrinsics(2, (11, 10), 3, 0.1)
    #     ntools.assert_false(ci1 == ci2)
    #     ntools.assert_true(ci1 != ci2)
