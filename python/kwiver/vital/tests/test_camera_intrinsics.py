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

from kwiver.vital.tests.cpp_helpers import camera_intrinsics_helpers
from kwiver.vital.tests.py_helpers import no_call_pure_virtual_method
from kwiver.vital.types import CameraIntrinsics, SimpleCameraIntrinsics

class TestCameraInstrinsicsBase(unittest.TestCase):
    def test_init(self):
        CameraIntrinsics()

    def test_virt_methods(self):
        ci = CameraIntrinsics()
        no_call_pure_virtual_method(ci.focal_length)
        no_call_pure_virtual_method(ci.principal_point)
        no_call_pure_virtual_method(ci.aspect_ratio)
        no_call_pure_virtual_method(ci.skew)
        no_call_pure_virtual_method(ci.image_width)
        no_call_pure_virtual_method(ci.image_height)
        no_call_pure_virtual_method(ci.map, np.array([0, 1]))
        no_call_pure_virtual_method(ci.unmap, np.array([0 ,1]))
        no_call_pure_virtual_method(ci.as_matrix)

    def test_members(self):
        ci = CameraIntrinsics()
        nt.ok_(ci.is_map_valid(np.array([2, 1, 1])))
        np.testing.assert_array_almost_equal(np.array([3, 2]), ci.distort(np.array([3, 2])))
        np.testing.assert_array_almost_equal(np.array([3, 2]), ci.undistort(np.array([3 ,2])))


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
        no_call_pure_virtual_method(SimpleCameraIntrinsics, CameraIntrinsics())
        SimpleCameraIntrinsics(SimpleCameraIntrinsics())

    def test_init_from_string(self):
        ret_str = "0.09 1 1 0.83 2 3 4 2 9"
        ret_intr = SimpleCameraIntrinsics.from_string(ret_str)
        nt.ok_(isinstance(ret_intr, SimpleCameraIntrinsics))
        nt.assert_equal(ret_intr.focal_length(), 0.09)
        np.testing.assert_array_almost_equal(ret_intr.principal_point(), np.array([1, 3]))

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

    def test_max_distort_radius(self):
        s = SimpleCameraIntrinsics(self.focal_length, self.principal_point, dist_coeffs=np.array([-0.1, 0.0, 0.0]),)
        a = s.max_distort_radius()*s.max_distort_radius()
        b = s.get_max_distort_radius_sq()
        nt.assert_equal(a, b)

    def dist_deriv(self, r, a, b, c):
        r2 = r*r
        return 1 + 3 * a *r2 + 5 * b * r2 * r2 + 7 * c * r2 * r2 * r2

    def check_finite_max_radius(self, a, b, c):
        mr = SimpleCameraIntrinsics().max_distort_radius_sq(a, b, c)
        self.assertGreater(mr, 0.0)
        nt.ok_(np.isfinite(mr))
        nt.assert_almost_equal(self.dist_deriv(mr, a, b, c), 0.0)

    def test_max_distort_radius_sq(self):
        self.check_finite_max_radius(0.0, -0.2, 0.0)
        self.check_finite_max_radius(-2.0/3, 1.0/5, 0.0)

    def test_str(self):
        s = SimpleCameraIntrinsics(self.focal_length, self.principal_point)
        s = str(s)
        nt.ok_(isinstance(s, str))

    def test_as_matrix(self):
        np.testing.assert_equal(SimpleCameraIntrinsics().as_matrix(), np.eye(3))
        np.testing.assert_equal(
            SimpleCameraIntrinsics(10, (2, 3), 2, 5).as_matrix(),
            [[10, 5, 2], [0, 5, 3], [0, 0, 1]],
        )
    def test_map_unmap(self):
        s = SimpleCameraIntrinsics(
            self.focal_length,
            self.principal_point,
            self.aspect_ratio,
            self.skew,
            self.dist_coeffs,
            self.image_width,
            self.image_height,
        )
        np.testing.assert_array_almost_equal(np.array([35377.05, 16426.53]), s.map(np.array([3, 2])))
        np.testing.assert_array_almost_equal(np.array([0.084889, -0.316774]), s.unmap(np.array([3 ,2])))

    def test_distort_undistort(self):
        s = SimpleCameraIntrinsics(
            self.focal_length,
            self.principal_point,
            self.aspect_ratio,
            self.skew,
            self.dist_coeffs,
            self.image_width,
            self.image_height,
        )
        np.testing.assert_array_almost_equal(np.array([2814.9, 1876.6]), s.distort(np.array([3, 2])))
        np.testing.assert_array_almost_equal(np.array([0.92024957, 0.61349972]), s.undistort(np.array([3 ,2])))

    def test_is_map_valid(self):
        s = SimpleCameraIntrinsics(
            self.focal_length,
            self.principal_point,
            self.aspect_ratio,
            self.skew,
            self.dist_coeffs,
            self.image_width,
            self.image_height,
        )
        nt.ok_(s.is_map_valid(np.array([1.0, -1.0])))
    def test_clone(self):
        s = SimpleCameraIntrinsics(self.focal_length, self.principal_point, dist_coeffs=np.array([-0.1, 0.0, 0.0]),)
        s_clone = s.clone()
        nt.ok_(isinstance(s_clone, SimpleCameraIntrinsics))
        nt.assert_equal(s.focal_length(), s_clone.focal_length())


class InheritedCameraIntrinsics(CameraIntrinsics):
    def __init__(self, focal_length, principal_point):
        CameraIntrinsics.__init__(self)
        self.focal_ = focal_length
        self.prin_pt = principal_point
        self.aspect_ratio_ = 1.2
        self.skew_ = 3.1
        self.dist_coeffs_ = [4.5, 5.2, 6.8]

    def clone(self):
        return InheritedCameraIntrinsics(self.focal_, self.prin_pt)

    def focal_length(self):
        return self.focal_

    def principal_point(self):
        return self.prin_pt

    def aspect_ratio(self):
        return self.aspect_ratio_

    def skew(self):
        return self.skew_

    def image_width(self):
        return 1080

    def image_height(self):
        return 720

    def dist_coeffs(self):
        return self.dist_coeffs_

    def as_matrix(self):
        return np.ndarray((3,3), dtype=float)

    def map(self, vec2):
        return vec2

    def unmap(self, vec2):
        return vec2

    def distort(self, vec2):
        return vec2

    def undistort(self, vec2):
        return vec2

    def is_map_valid(self, vec3):
        return True

class TestInheritedCamIntrins(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.focal_length = 10.5
        self.principal_point = [3.14, 6.28]
        self.aspect_ratio = 1.2
        self.skew = 3.1
        self.dist_coeffs = [4.5, 5.2, 6.8]
        self.image_width = 1080
        self.image_height = 720
        self.inCam = InheritedCameraIntrinsics(self.focal_length, self.principal_point)

    def test_subclass(self):
        nt.ok_(issubclass(InheritedCameraIntrinsics, CameraIntrinsics))

    def test_clone(self):
        cloned = camera_intrinsics_helpers.clone(self.inCam)
        nt.ok_(isinstance(cloned, InheritedCameraIntrinsics))
        nt.assert_equal(cloned.focal_length(), self.inCam.focal_length())

    def test_fl(self):
        fl = camera_intrinsics_helpers.focal_length(self.inCam)
        self.assertEquals(self.focal_length, fl)

    def test_principal_point(self):
        p = camera_intrinsics_helpers.principal_point(self.inCam)
        np.testing.assert_array_equal(p, self.principal_point)

    def test_aspect_ratio(self):
        ar = camera_intrinsics_helpers.aspect_ratio(self.inCam)
        np.testing.assert_array_equal(ar, self.aspect_ratio)

    def test_skew(self):
        s = camera_intrinsics_helpers.skew(self.inCam)
        nt.assert_equal(s, self.skew)

    def test_image_width_height(self):
        w = camera_intrinsics_helpers.image_width(self.inCam)
        h = camera_intrinsics_helpers.image_height(self.inCam)
        nt.assert_equal(w, self.image_width)
        nt.assert_equal(h, self.image_height)

    def test_dist_coeffs(self):
        dist_co = camera_intrinsics_helpers.dist_coeffs(self.inCam)
        np.testing.assert_array_equal(dist_co, self.dist_coeffs)

    def test_as_matrix(self):
        as_mat = camera_intrinsics_helpers.as_matrix(self.inCam)
        np.testing.assert_array_equal(as_mat, np.ndarray((3,3), dtype=float))

    def test_map_unmap(self):
        np.testing.assert_array_equal(camera_intrinsics_helpers.map
                                    (self.inCam, np.array([2, 3])), np.array([2, 3]))
        np.testing.assert_array_equal(camera_intrinsics_helpers.unmap(
                                     self.inCam, np.array([2, 3])),
                                     np.array([2, 3]))

    def test_distort_undistort(self):
        np.testing.assert_array_equal(camera_intrinsics_helpers.distort
                                    (self.inCam, np.array([2, 3])), np.array([2, 3]))
        np.testing.assert_array_equal(camera_intrinsics_helpers.undistort(
                                     self.inCam, np.array([2, 3])),
                                     np.array([2, 3]))

    def test_is_map_valid(self):
        nt.ok_(camera_intrinsics_helpers.is_map_valid(self.inCam, np.array([3, 2, 1])))
