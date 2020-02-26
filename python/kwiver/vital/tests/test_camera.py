"""
ckwg +31
Copyright 2016-2017 by Kitware, Inc.
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

Tests for Camera interface class.

"""
from __future__ import print_function
import math
import unittest
from numpy.random import rand
import nose.tools
import pytest
import numpy
import os

from kwiver.vital.types import (
    Camera,
    CameraIntrinsics,
    EigenArray,
    Rotation,
)

@pytest.mark.skip(reason="Old binding based tests that are being updated")
class TestVitalCamera (unittest.TestCase):

    def test_new(self):
        # just seeing that basic construction doesn't blow up
        cam = Camera()

        c = EigenArray(3)
        r = Rotation()
        ci = CameraIntrinsics()
        cam = Camera(c, r, ci)

    def test_center_default(self):
        cam = Camera()
        numpy.testing.assert_array_equal(
            cam.center,
            [0,0,0]
        )

    def test_center_initialized(self):
        expected_c = EigenArray(3)
        expected_c.get_matrix()[:] = [[1],[2],[3]]
        cam = Camera(expected_c)
        numpy.testing.assert_array_equal(
            numpy.array([cam.center]).T,
            expected_c.get_matrix()
        )
        numpy.testing.assert_array_equal(
            cam.center,
            [1,2,3]
        )

    def test_translation_default(self):
        cam = Camera()
        numpy.testing.assert_array_equal(
            cam.translation,
            [0, 0, 0]
        )

    def test_translation_initialized(self):
        center = EigenArray.from_array([[1],[2],[3]])
        rotation = Rotation.from_axis_angle([0, 1, 0], math.pi / 2.)
        cam = Camera(center, rotation)
        numpy.testing.assert_array_equal(
            cam.translation,
            -(rotation * center.get_matrix())
        )

    def test_covariance_default(self):
        cam = Camera()
        # Should be default value of an identity matrix
        numpy.testing.assert_array_equal(
            cam.covariance.to_matrix(),
            numpy.eye(3),
        )

    def test_rotation_default(self):
        cam = Camera()
        # Should be identity by default
        numpy.testing.assert_array_equal(
            cam.rotation.matrix(),
            numpy.eye(3),
        )

    def test_rotation_initialized(self):
        center = EigenArray.from_array([[1],[2],[3]])
        r_expected = Rotation.from_axis_angle([0,1,0], math.pi / 8)
        cam = Camera(center, r_expected)
        nose.tools.assert_is_not(cam.rotation, r_expected)
        nose.tools.assert_equal(cam.rotation, r_expected)

    def test_asmatrix_default(self):
        cam = Camera()
        m_expected = numpy.matrix("1 0 0 0;"
                                  "0 1 0 0;"
                                  "0 0 1 0")
        print(cam.as_matrix())
        print(m_expected)
        numpy.testing.assert_array_equal(
            cam.as_matrix(),
            m_expected
        )

    def test_depth(self):
        cam = Camera()

        pos_pt = numpy.array([1,0,1])
        neg_pt = numpy.array([0,0,-100])

        nose.tools.assert_equal(cam.depth(pos_pt), 1)
        nose.tools.assert_equal(cam.depth(neg_pt), -100)

    def test_equal(self):
        cam1 = Camera()
        cam2 = Camera()
        nose.tools.assert_equal(cam1, cam1)
        nose.tools.assert_equal(cam1, cam2)

        center = EigenArray.from_array([[1],[2],[3]])
        rotation = Rotation.from_axis_angle([0, 1, 0], math.pi / 2.)
        cam1 = Camera(center, rotation)
        cam2 = Camera(center, rotation)
        nose.tools.assert_equal(cam1, cam1)
        nose.tools.assert_equal(cam1, cam2)

    def test_to_from_string(self):
        cam = Camera()
        cam_s = cam.as_string()
        cam2 = Camera.from_string(cam_s)
        print("Default camera string:\n%s" % cam_s)
        print("Default newcam string:\n%s" % cam2.as_string())
        nose.tools.assert_equal(cam, cam2)

        center = EigenArray.from_array([[1],
                                        [2],
                                        [3]])
        rotation = Rotation.from_axis_angle([0, 1, 0], math.pi / 2.)
        cam = Camera(center, rotation)
        cam_s = cam.as_string()
        cam2 = Camera.from_string(cam_s)
        print("Custom camera string:\n%s" % cam_s)
        print("Custom newcam string:\n%s" % cam2.as_string())
        nose.tools.assert_equal(cam, cam2)

    def test_clone_look_at(self):
        pp = EigenArray.from_array([[300], [400]])
        k = CameraIntrinsics(1000, [300, 400])
        focus = EigenArray.from_array([[0], [1], [-2]])
        center = EigenArray.from_array([[3],[-4],[7]])

        base = Camera(center, Rotation(), k)
        cam = base.clone_look_at(numpy.array([0,1,2]))
        nose.tools.assert_not_equal(base, cam)

        ifocus = cam.project([0,1,2])
        nose.tools.assert_almost_equal(numpy.linalg.norm(ifocus - pp.get_matrix().T, 2),
                                       0., 12)

        ifocus_up = cam.project([0,1,4])
        tmp = (ifocus_up - pp.get_matrix().T)[0]
        nose.tools.assert_almost_equal(tmp[0], 0., 12)
        nose.tools.assert_true(tmp[1] < 0.)

    def test_read_write_krtd_file(self):
        # Use a random string filename to avoid name collision.
        fname = 'temp_camera_test_read_write_krtd_file.txt'

        try:
            for _ in range(100):
                c = (rand(3)*2-1)*100
                center = EigenArray.from_array([c])
                rotation = Rotation.from_quaternion(numpy.random.rand(4)*2-1)
                intrinsics = CameraIntrinsics(10, (5, 5), 1.2, 0.5, [4, 5, 6])
                c1 = Camera(center, rotation,
                            intrinsics)

                c1.write_krtd_file(fname)
                c2 = Camera.from_krtd_file(fname)

                err = numpy.linalg.norm(c1.center-c2.center)
                assert err < 1e-9, ''.join(['Centers are different by ',
                                            str(err)])

                c1.rotation.angle_from(c2.rotation) < 1e-12

                attr = ['focal_length','aspect_ratio','principle_point','skew',
                        'dist_coeffs']
                for att in attr:
                    v1 = numpy.array(getattr(c1.intrinsics,att))
                    v2 = numpy.array(getattr(c2.intrinsics,att))
                    err = numpy.linalg.norm(v1-v2)
                    assert err < 1e-8, ''.join(['Difference ',str(err),
                                                 ' for attribute: ',att])
        finally:
            if os.path.isfile(fname):
                os.remove(fname)
