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

Tests for Python interface to vital::local_cartesian

"""

from kwiver.vital.types import LocalCartesian, GeoPoint, geodesy
from kwiver.vital.modules import modules

import nose.tools as nt
import numpy as np


class TestVitalLocalCartesian(object):
    def setUp(self):
        self.wgs = geodesy.SRID.lat_lon_WGS84
        self.crs_utm_18n = geodesy.SRID.UTM_WGS84_north + 18

        self.origA = GeoPoint(np.array([-73.759291, 42.849631, 0]), self.wgs)
        self.origB = GeoPoint(np.array([601375.01, 4744863.31, 0]), self.crs_utm_18n)
        self.offset1 = np.array([25, 55, 0])
        self.offset2 = np.array([250, 5500, 50])
        self.geo1 = GeoPoint(np.array([-73.75898515, 42.85012609, 0]), self.wgs)
        self.geo2 = GeoPoint(np.array([-73.75623008, 42.89913984, 52.381]), self.wgs)

    def test_create(self):
        LocalCartesian(self.origA, 0)
        LocalCartesian(self.origA, 33)

    def test_initial_origin(self):
        lc = LocalCartesian(self.origA, 0)
        np.testing.assert_array_almost_equal(lc.get_origin().location(), self.origA.location())

        lc = LocalCartesian(self.origA, 33)
        np.testing.assert_array_almost_equal(lc.get_origin().location(), self.origA.location())

    def test_initial_orientation(self):
        lc = LocalCartesian(self.origA, 0)
        nt.assert_equal(lc.get_orientation(), 0)

        lc = LocalCartesian(self.origA, 33)
        nt.assert_equal(lc.get_orientation(), 33)

    def check_origin_and_orient_equal(self, lc, origin, orient):
        np.testing.assert_array_almost_equal(lc.get_origin().location(), origin.location())
        nt.assert_equal(lc.get_orientation(), orient)

    def test_api(self):
        lc = LocalCartesian(self.origA, 0)

        lc.set_origin(self.geo1, 33)
        self.check_origin_and_orient_equal(lc, self.geo1, 33)

        lc.set_origin(self.geo2, 22)
        self.check_origin_and_orient_equal(lc, self.geo2, 22)

    def test_default_args(self):
        # Default value of orientation arg = 0
        lc = LocalCartesian(self.origA)
        self.check_origin_and_orient_equal(lc, self.origA, 0)

        # Test calling set_origin with named parameters
        lc.set_origin(orientation=33, origin=self.geo1)
        self.check_origin_and_orient_equal(lc, self.geo1, 33)

        # Now test named parameters of constructors
        lc = LocalCartesian(orientation=33, origin=self.origA)
        self.check_origin_and_orient_equal(lc, self.origA, 33)

        # Then default value of orientation argument for set_origin
        lc.set_origin(self.geo1)
        self.check_origin_and_orient_equal(lc, self.geo1, 0)

    def compare_lla(self, gp1, gp2):
        np.testing.assert_almost_equal(gp1[0], gp2[0], 7)
        np.testing.assert_almost_equal(gp1[1], gp2[1], 7)
        np.testing.assert_almost_equal(gp1[2], gp2[2], 3)

    def test_conversion(self):
        modules.load_known_modules()

        geo_outA = GeoPoint()
        lc_lla = LocalCartesian(self.origA)

        lc_lla.convert_from_cartesian(self.offset1, geo_outA)
        self.compare_lla(geo_outA.location(), self.geo1.location())

        cart_outA = lc_lla.convert_to_cartesian(geo_outA)
        self.compare_lla(cart_outA, self.offset1)

        lc_lla.convert_from_cartesian(self.offset2, geo_outA)
        self.compare_lla(geo_outA.location(), self.geo2.location())

        cart_outA = lc_lla.convert_to_cartesian(geo_outA)
        self.compare_lla(cart_outA, self.offset2)

        geo_outB = GeoPoint()
        lc_utm = LocalCartesian(self.origB)

        lc_utm.convert_from_cartesian(self.offset1, geo_outB)
        self.compare_lla(geo_outB.location(), self.geo1.location())

        cart_outB = lc_utm.convert_to_cartesian(geo_outB)
        self.compare_lla(cart_outB, self.offset1)

        lc_utm.convert_from_cartesian(self.offset2, geo_outB)
        self.compare_lla(geo_outB.location(), self.geo2.location())

        cart_outB = lc_utm.convert_to_cartesian(geo_outB)
        self.compare_lla(cart_outB, self.offset2)
