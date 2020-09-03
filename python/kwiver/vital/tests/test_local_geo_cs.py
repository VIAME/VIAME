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

Tests for Python interface to vital::local_geo_cs

"""


import nose.tools as nt
import unittest
import numpy as np
from kwiver.vital.modules import modules
from kwiver.vital.types import (
    LocalGeoCS,
    GeoPoint,
    geodesy,
    local_geo_cs,
)

modules.load_known_modules()

class TestLocalGeoCS(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.wgs = geodesy.SRID.lat_lon_WGS84
        self.geo1 = GeoPoint(np.array([-73.75898515, 42.85012609, 0]), self.wgs)
        self.geo2 = GeoPoint(np.array([-73.75623008, 42.89913984, 52.381]), self.wgs)
    def test_init(self):
        LocalGeoCS()
    def test_origin(self):
        g = LocalGeoCS()
        g.geo_origin = self.geo1
        np.testing.assert_array_almost_equal(g.geo_origin.location(self.wgs), self.geo1.location())
    def test_read_write_localgeocs(self):
        g = LocalGeoCS()
        g.geo_origin = self.geo1
        f = open("geo_data.txt", "w+")
        f.close()
        local_geo_cs.write_local_geo_cs_to_file(g, "geo_data.txt")
        g2 = LocalGeoCS()
        local_geo_cs.read_local_geo_cs_from_file(g2, "geo_data.txt")
        np.testing.assert_array_almost_equal(g2.geo_origin.location(self.wgs), self.geo1.location())
