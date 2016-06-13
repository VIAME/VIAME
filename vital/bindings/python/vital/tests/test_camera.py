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

Tests for Camera interface class.

"""
import math
import unittest

import nose.tools
import numpy

from vital.types import (
    Camera,
    CameraIntrinsics,
    EigenArray,
    Rotation,
)


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
            [[0],
             [0],
             [0]]
        )

    def test_center_initialized(self):
        expected_c = EigenArray(3)
        expected_c[:] = [[1],
                         [2],
                         [3]]
        cam = Camera(expected_c)
        numpy.testing.assert_array_equal(
            cam.center,
            expected_c
        )
        numpy.testing.assert_array_equal(
            cam.center,
            [[1],
             [2],
             [3]]
        )

    def test_translation_default(self):
        cam = Camera()
        numpy.testing.assert_array_equal(
            cam.translation,
            [[0], [0], [0]]
        )

    def test_translation_initialized(self):
        center = EigenArray.from_iterable([[1],
                                           [2],
                                           [3]])
        rotation = Rotation.from_axis_angle([[0], [1], [0]], math.pi / 2.)
        cam = Camera(center, rotation)
        numpy.testing.assert_array_equal(
            cam.translation,
            -(rotation * center)
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
        r_expected = Rotation.from_axis_angle([[0],[1],[0]], math.pi / 8)
        cam = Camera(rotation=r_expected)
        nose.tools.assert_is_not(cam.rotation, r_expected)
        nose.tools.assert_equal(cam.rotation, r_expected)

    def test_asmatrix_default(self):
        cam = Camera()
        m_expected = numpy.matrix("1 0 0 0;"
                                  "0 1 0 0;"
                                  "0 0 1 0")
        print cam.as_matrix()
        print m_expected
        numpy.testing.assert_array_equal(
            cam.as_matrix(),
            m_expected
        )

    def test_depth(self):
        cam = Camera()

        pos_pt = [[1],
                  [0],
                  [1]]
        neg_pt = [[0],
                  [0],
                  [-100]]

        nose.tools.assert_equal(cam.depth(pos_pt), 1)
        nose.tools.assert_equal(cam.depth(neg_pt), -100)

    def test_equal(self):
        cam1 = Camera()
        cam2 = Camera()
        nose.tools.assert_equal(cam1, cam1)
        nose.tools.assert_equal(cam1, cam2)

        center = EigenArray.from_iterable([[1],
                                           [2],
                                           [3]])
        rotation = Rotation.from_axis_angle([[0], [1], [0]], math.pi / 2.)
        cam1 = Camera(center, rotation)
        cam2 = Camera(center, rotation)
        nose.tools.assert_equal(cam1, cam1)
        nose.tools.assert_equal(cam1, cam2)

    def test_to_from_string(self):
        cam = Camera()
        cam_s = cam.as_string()
        cam2 = Camera.from_string(cam_s)
        print "Default camera string:\n%s" % cam_s
        print "Default newcam string:\n%s" % cam2.as_string()
        nose.tools.assert_equal(cam, cam2)

        center = EigenArray.from_iterable([[1],
                                           [2],
                                           [3]])
        rotation = Rotation.from_axis_angle([[0], [1], [0]], math.pi / 2.)
        cam = Camera(center, rotation)
        cam_s = cam.as_string()
        cam2 = Camera.from_string(cam_s)
        print "Custom camera string:\n%s" % cam_s
        print "Custom newcam string:\n%s" % cam2.as_string()
        nose.tools.assert_equal(cam, cam2)
