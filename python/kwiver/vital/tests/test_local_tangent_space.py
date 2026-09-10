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

Tests for Python interface to vital::local_tangent_space

"""

import unittest
import numpy as np
import os

from kwiver.vital.types import (
    LocalTangentSpace,
    GeoPoint,
    RotationD,
    geodesy,
    read_local_tangent_space_from_file,
    write_local_tangent_space_to_file,
)
from kwiver.vital import plugin_management


class TestLocalTangentSpace(unittest.TestCase):
    @classmethod
    def setUp(self):
        vpm = plugin_management.plugin_manager_instance()
        vpm.load_all_plugins()
        self.wgs = geodesy.SRID.lat_lon_WGS84
        self.geo1 = GeoPoint(np.array([-73.75898515, 42.85012609, 0]), self.wgs)
        self.geo2 = GeoPoint(np.array([-73.75623008, 42.89913984, 52.381]), self.wgs)
        self.rot = RotationD(1, 2, 3)

    def test_init(self):
        LocalTangentSpace()
        LocalTangentSpace(self.geo1)

    def test_valid(self):
        assert not LocalTangentSpace().valid()
        assert LocalTangentSpace(self.geo1).valid()

    def test_origin(self):
        g = LocalTangentSpace(self.geo1)
        np.testing.assert_array_almost_equal(
            g.origin.location(self.wgs), self.geo1.location()
        )

    def test_point(self):
        g = LocalTangentSpace(self.geo1)
        p = g.to_global(g.to_local(self.geo2))
        np.testing.assert_array_almost_equal(
            p.location(self.wgs), self.geo2.location(self.wgs)
        )

    def test_rotation(self):
        g = LocalTangentSpace(self.geo1)
        r = g.to_global(g.to_local(self.rot, self.geo2), self.geo2)
        np.testing.assert_array_almost_equal(
            np.array(r.yaw_pitch_roll()), np.array(self.rot.yaw_pitch_roll())
        )

    def test_read_write(self):
        filename = "local_space.txt"
        g = LocalTangentSpace(self.geo1)
        write_local_tangent_space_to_file(g, filename)
        g2 = read_local_tangent_space_from_file(filename)
        np.testing.assert_array_almost_equal(
            g.to_local(self.geo2), g2.to_local(self.geo2)
        )
        os.remove(filename)
