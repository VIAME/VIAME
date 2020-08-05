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

Tests for Camera Perspective python class.

"""

import unittest

import nose.tools as nt
import numpy as np
from kwiver.vital.tests.py_helpers import no_call_pure_virtual_method
from kwiver.vital.types import CameraPerspective as cap
from kwiver.vital.types import SimpleCameraPerspective as scap
from kwiver.vital.tests.cpp_helpers import camera_perspective_helpers as cph
from kwiver.vital.types import (
    Camera,
    SimpleCameraIntrinsics as ci,
    rotation,
    RotationD,
)
from kwiver.vital.types.covariance import *

# Test python classes inherited from C++ w/ virtual methods overridden
class CameraPerspectiveImpl(cap):
    def __init__(self, rot_, cen_, intrin_):
        cap.__init__(self)
        self.rot = rot_
        self.center = cen_
        self.intrins = intrin_

    def clone(self):
        return CameraPerspectiveImpl(self.rot, self.center, self.intrins)
    def get_center(self):
        return self.center
    def translation(self):
        return -(self.rot.inverse() * self.center)
    def center_covar(self):
        return Covar3d()
    def rotation(self):
        return self.rot
    def intrinsics(self):
        return self.intrins
    def image_width(self):
        return self.intrins.image_width()
    def image_height(self):
        return self.intrins.image_height()
    def as_matrix(self):
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
    def clone_look_at(self, stare_point, up):
        return CameraPerspectiveImpl(RotationD(0, [0, 1, 0]), self.center, self.intrins)
    def pose_matrix(self):
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
    def project(self, pt):
        return pt[:2]
    def depth(self, pt):
        return pt[2:]


# Test CameraPerspective
class TestCameraPerspective(unittest.TestCase):
    def test_construct(self):
        cap()

    # Test pure virtual methods
    # to ensure calls cannot be made to a pure virtual method
    # Note: clone is skipped.
    def test_pure_virtual_methods(self):
        no_call_pure_virtual_method(cap().center)
        no_call_pure_virtual_method(cap().translation)
        no_call_pure_virtual_method(cap().center_covar)
        no_call_pure_virtual_method(cap().rotation)
        no_call_pure_virtual_method(cap().intrinsics)
    # Test that the virtual methods with default impls
    # dependent on pure virtual methods also cannot be called
    def test_non_pure_virtual_methods(self):
        no_call_pure_virtual_method(cap().depth, np.array([1, 1, 1]))
        no_call_pure_virtual_method(cap().project, np.array([1, 2, 3]))
        no_call_pure_virtual_method(cap().as_matrix)
        #pose matrix is not virtual, but it's impl does call pure virts
        no_call_pure_virtual_method(cap().pose_matrix)
        no_call_pure_virtual_method(cap().image_width)
        no_call_pure_virtual_method(cap().image_height)

# Test simple Camera Perspective, impl of Camera Perspective
class TestSimpleCameraPerspective(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.intrins = ci(focal_length=10.5,
            principal_point=[3.14, 6.28],
            aspect_ratio=1.2,
            skew=3.1,
            dist_coeffs=[4.5, 5.2, 6.8],
            image_width=1080,
            image_height=720,
        )
        self.intrins_empty = ci()
        self.rot = RotationD(0, [1, 0, 0])
        self.vec = np.array([4, 3.14, 5.67])

        m = np.eye(3, dtype=np.double)
        m[0, 2] = 2.0
        m_expected = m.copy()
        m_expected[0, 2] = 1.0
        m_expected[2, 0] = 1.0
        self.covar3d = Covar3d(m)

    def test_construct(self):
        scap()
        scap(center=self.vec, rotation=self.rot, intrinsics=self.intrins)
        scap(center=self.vec, rotation=self.rot)
        scap(scap())

    def test_overridden(self):
        a = scap(center=self.vec, rotation=self.rot, intrinsics=self.intrins)
        clone_ = cph.call_clone(a)
        self.assertIsInstance(clone_, scap)
        np.testing.assert_array_equal(clone_.get_center(), self.vec)
        center_ = cph.call_center(a)
        np.testing.assert_array_equal(center_, self.vec)
        trans_ = cph.call_translation(a)
        np.testing.assert_array_equal(trans_, -(a.get_rotation().inverse()*a.get_center()))
        covar_ = cph.call_center_covar(a)
        self.assertIsInstance(covar_, Covar3d)
        np.testing.assert_array_equal(covar_.matrix(), np.array([[1, 0, 0],
                                                                    [0, 1, 0],
                                                                    [0, 0, 1]]))
        rot_ = cph.call_rotation(a)
        nt.assert_equal(rot_, self.rot)
        intr_ = cph.call_intrinsics(a)
        nt.assert_equal(intr_, self.intrins)
        ih_ = cph.call_image_height(a)
        self.assertEqual(ih_, 720)
        iw_ = cph.call_image_width(a)
        self.assertEqual(iw_, 1080)
        clone_look_at_ = cph.call_clone_look_at(a, np.array([1, 0, 0]), np.array([0,1,0]))
        self.assertIsInstance(clone_look_at_, scap)
        nt.assert_not_equal(clone_look_at_.get_rotation(), self.rot)
        cam_ = cph.call_as_matrix(a)
        np.testing.assert_array_almost_equal(cam_, np.array([[10.5, 3.1, 3.14, -69.5378],
                                                        [0, 8.75, 6.28, -63.0826],
                                                        [0, 0, 1, -5.67]]
                                                        ))



    def test_not_overidden(self):
        # Test getters
        scap_ = scap(center=self.vec, rotation=self.rot, intrinsics=self.intrins)
        np.testing.assert_array_equal(scap_.get_center(), self.vec)
        self.assertIsInstance(scap_.get_rotation(), RotationD)
        nt.assert_equal(scap_.get_rotation(),self.rot)
        self.assertIsInstance(scap_.get_intrinsics(), ci)
        nt.assert_equal(scap_.get_intrinsics(), self.intrins)

        # Test setters
        new_center = np.array([0, 2, 0])
        scap_.set_center(new_center)
        np.testing.assert_array_equal(scap_.get_center(),new_center)
        new_rot = RotationD(1, np.array([0, 2, 0]))
        scap_.set_rotation(new_rot)
        nt.assert_equal(scap_.get_rotation(), new_rot)
        scap_.set_intrinsics(self.intrins_empty)
        nt.assert_equal(scap_.get_intrinsics(), self.intrins_empty)
        scap_.set_translation(np.array([1, 0, 3]))
        np.testing.assert_array_equal(scap_.get_center(), -(scap_.get_rotation().inverse()*np.array([1, 0, 3])))

        # Test covar setter/gettter
        m_out = self.covar3d.matrix()
        scap_.set_center_covar(self.covar3d)
        np.testing.assert_array_equal(scap_.get_center_covar().matrix(), m_out)

        # Test other (look_at)
        scap_= scap()
        scap_.look_at(np.array([1, 0, 0]), np.array([0, 0, 1]))
        rot_ = scap_.get_rotation()
        np.testing.assert_array_almost_equal(rot_.matrix(), np.array([[0, -1, 0],
                                                             [0, 0, -1],
                                                             [1, 0 ,0]]))

class TestCameraPerspectiveImpl(unittest.TestCase):
    def test_init(self):
        rot_ = RotationD(0, [1, 0, 0])
        intrins_ = ci(focal_length=10.5,
            principal_point=[3.14, 6.28],
            aspect_ratio=1.2,
            skew=3.1,
            dist_coeffs=[4.5, 5.2, 6.8],
            image_width=1080,
            image_height=720,
        )

        cent = np.array([4, 3.14, 5.67])
        CameraPerspectiveImpl(rot_, cent, intrins_)
    def test_inheritance(self):
        nt.ok_(issubclass(CameraPerspectiveImpl,cap))

    def test_clone_clone_look_at(self):
        rot_ = RotationD(0, [1, 0, 0])
        intrins_ = ci(focal_length=10.5,
            principal_point=[3.14, 6.28],
            aspect_ratio=1.2,
            skew=3.1,
            dist_coeffs=[4.5, 5.2, 6.8],
            image_width=1080,
            image_height=720,
        )

        cent = np.array([4, 3.14, 5.67])
        cam_test = CameraPerspectiveImpl(rot_, cent, intrins_)
        campp = cph.call_clone(cam_test)
        self.assertIsInstance(campp, CameraPerspectiveImpl)
        np.testing.assert_array_equal(campp.center, cent)

        campp_look_at = cph.call_clone_look_at(cam_test, np.array([2, 3, 4]), np.array([0, 0, 1]))
        self.assertIsInstance(campp_look_at, CameraPerspectiveImpl)
        nt.assert_equal(campp_look_at.rot, RotationD(0, [0, 1, 0]))

    def test_overrides(self):
        rot_ = RotationD(0, [1, 0, 0])
        intrins_ = ci(focal_length=10.5,
            principal_point=[3.14, 6.28],
            aspect_ratio=1.2,
            skew=3.1,
            dist_coeffs=[4.5, 5.2, 6.8],
            image_width=1080,
            image_height=720,
        )

        cent = np.array([4, 3.14, 5.67])
        cam_test = CameraPerspectiveImpl(rot_, cent, intrins_)

        np.testing.assert_array_equal(cam_test.get_center(), cent)
        nt.assert_equal(cam_test.rotation(), rot_)
        np.testing.assert_array_equal(cam_test.translation(), -(cam_test.rot.inverse() * cam_test.center))
        np.testing.assert_array_equal(cam_test.center_covar().matrix(), np.array([[1, 0, 0],
                                                                                  [0, 1, 0],
                                                                                  [0, 0, 1]]))
        nt.assert_equal(cam_test.intrinsics(), intrins_)
        self.assertEqual(cam_test.image_height(), 720)
        self.assertEqual(cam_test.image_width(), 1080)
        np.testing.assert_array_equal(cam_test.pose_matrix(), np.array([[1, 0, 0],
                                                                        [0, 1, 0],
                                                                        [0, 0, 1]]))
        np.testing.assert_array_equal(cam_test.as_matrix(), np.array([[1, 0, 0],
                                                                        [0, 1, 0],
                                                                        [0, 0, 1]]))
        pt = np.array([1, 2, 3])
        np.testing.assert_array_equal(cam_test.project(pt), pt[:2])
        np.testing.assert_array_equal(cam_test.depth(pt), pt[2:])
