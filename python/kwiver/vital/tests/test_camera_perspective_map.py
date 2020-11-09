"""
ckwg +31
Copyright 2020 by Kitware, Inc.
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

Tests for CameraPerspectiveMap interface

"""

import unittest
import nose.tools as nt
import numpy as np
from kwiver.vital.types import CameraMap
from kwiver.vital.types import CameraPerspectiveMap as cam
from kwiver.vital.types import SimpleCameraPerspective as scap
from kwiver.vital.tests.cpp_helpers import camera_perspective_map_helpers as cpmh


class CameraPerspectiveMapTest(unittest.TestCase):
    def setUp(self):
        self.a = scap()
        self.a2 = scap()
        self.a3 = scap()
        self.b = {1:self.a,2:self.a2,3:self.a3}
        self.ca = cam(self.b)
    def test_constructors(self):
        cam()
        a = scap()
        b = {1:a}
        cam(b)
    def test_size(self):
        # size()
        self.assertEqual(self.ca.size(),3)
    def test_cameras(self):
        # cameras()
        ret_dict = self.ca.cameras()
        self.assertIsInstance(ret_dict, dict)
        self.assertEqual(len(ret_dict),3)
        nt.assert_equal(ret_dict[1],self.a)
    def test_frame_ids(self):
        # get_frame_ids()
        ret_set = self.ca.get_frame_ids()
        self.assertIsInstance(ret_set, set)
        self.assertEqual(len(ret_set), 3)
        self.assertSetEqual(ret_set,{1,2,3})
    def test_find(self):
        # find
        ret_persp = self.ca.find(2)
        self.assertIsInstance(ret_persp, scap)
        nt.assert_equal(ret_persp, self.a2)
    def test_erase(self):
        # erase
        self.ca.erase(1)
        self.assertEqual(self.ca.size(), 2)
        self.assertEqual(len(self.ca.cameras()), 2)
        self.assertDictEqual(self.ca.cameras(), {2:self.a2, 3:self.a3})
    def test_insert(self):
        # insert
        self.ca.insert(1, self.a)
        self.assertDictEqual(self.b, self.ca.cameras())
        self.assertEqual(self.ca.size(), 3)
    def test_clone(self):
        # clone
        new_ca = self.ca.clone()
        self.assertIsInstance(new_ca, cam)
        self.assertEqual(new_ca.size(), 3)
        nt.assert_equal(new_ca.cameras().keys(), self.ca.cameras().keys())
    def test_clear(self):
        # clear
        self.ca.clear()
        self.assertEqual(self.ca.size(), 0)
        self.assertEqual(len(self.ca.cameras()), 0)
    def test_set_from_base_camera_map(self):
        # set_from_base_camera_map
        self.ca.set_from_base_camera_map(self.b)
        self.assertEqual(self.ca.size(), 3)
        self.assertDictEqual(self.ca.cameras(), self.b)

class CameraPerspectiveInheritance(cam):
    def __init__(self, cam_dict_):
        cam.__init__(self)
        self.cam_dict = cam_dict_

    def size(self):
        return len(self.cam_dict)
    def cameras(self):
        return self.cam_dict
    def get_frame_ids(self):
        return set(self.cam_dict.keys())

class TestCamPerspectiveInheritance(unittest.TestCase):
    def test_construct(self):
        a1 = scap()
        a2 = scap()
        cam_dict = {1:a1, 2:a2}
        CameraPerspectiveInheritance(cam_dict)
    def test_inheritance(self):
        a1 = scap()
        a2 = scap()
        cam_dct = {1:a1, 2:a2}
        CameraPerspectiveInheritance(cam_dct)
        nt.ok_(issubclass(CameraPerspectiveInheritance, cam))
    def test_methods(self):
        a1 = scap()
        a2 = scap()
        cam_dict = {1:a1, 2:a2}
        a = CameraPerspectiveInheritance(cam_dict)
        ret_size = cpmh.call_size(a)
        self.assertEqual(ret_size, 2)
        ret_cam_dict = cpmh.call_cameras(a)
        self.assertDictEqual(ret_cam_dict, cam_dict)
        a = CameraPerspectiveInheritance(cam_dict)
        ret_set = cpmh.call_get_frame_ids(a)
        self.assertSetEqual(ret_set, {1,2})
