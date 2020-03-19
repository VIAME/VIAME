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

Tests for Camera interface class.

"""
import unittest
import nose.tools as nt
import numpy as np

from kwiver.vital.tests.py_helpers import no_call_pure_virtual_method
from kwiver.vital.tests.cpp_helpers import camera_helpers as helper
from kwiver.vital.types import Camera


class SimpleCamera(Camera):
    def __init__(self, id_):
        Camera.__init__(self)
        self.id_ = id_

    def clone(self):
        return SimpleCamera(self.id_)

    def project(self, pt):
        return pt[:2]

    def image_width(self):
        return 1080

    def image_height(self):
        return 720

class TestVitalCameraSubclass(unittest.TestCase):
    def test_init(self):
        SimpleCamera(1)

    def test_inheritance(self):
        nt.ok_(issubclass(SimpleCamera, Camera))

    def test_clone_override(self):
        cam = SimpleCamera(2)
        cloned_cam = helper.call_clone(cam)

        # Check that the clone was not sliced
        nt.ok_(isinstance(cloned_cam, SimpleCamera))

        # Check ID is the same
        nt.assert_equal(cam.id_, cloned_cam.id_)

    def test_project_override(self):
        cam = SimpleCamera(3)
        pt = np.array([5, 10, 15])

        projected = helper.call_project(cam, pt)
        np.testing.assert_array_equal(projected, pt[:2])

    def test_width_override(self):
        cam = SimpleCamera(4)
        nt.assert_equal(helper.call_image_width(cam), 1080)

    def test_height_override(self):
        cam = SimpleCamera(4)
        nt.assert_equal(helper.call_image_height(cam), 720)

class TestVitalCamera(unittest.TestCase):
    def test_init(self):
        Camera()

    # Note that clone is skipped. See the binding file for an explanation

    def test_no_call_virtul_project(self):
        no_call_pure_virtual_method(Camera().project, np.array([-3.14, 3.14, 6.28]))

    def test_no_call_virtul_image_width(self):
        no_call_pure_virtual_method(Camera().image_width)

    def test_no_call_virtul_image_height(self):
        no_call_pure_virtual_method(Camera().image_height)
