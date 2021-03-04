# """
# ckwg +31
# Copyright 2020 by Kitware, Inc.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ==============================================================================

# Tests for Python interface to vital geodesy function/classes

# """
import nose.tools as nt
import numpy as np

from kwiver.vital.modules import modules
from kwiver.vital.types import geodesy as g
from kwiver.vital.tests.py_helpers import no_call_pure_virtual_method

class TestVitalGeodesy(object):
    def setUp(self):
        self.loc1 = np.array([-73.759291, 42.849631])
        self.loc2 = np.array([4.857878, 45.777158])
        self.loc3 = np.array([-62.557243, 82.505337])
        self.loc4 = np.array([-12.150267, 85.407630])
        self.loc5 = np.array([166.644316, -77.840078])
        self.loc6 = np.array([107.646964, -83.921037])
        self.pts = [self.loc1, self.loc2, self.loc3, self.loc4, self.loc5, self.loc6]

    def test_srid(self):
        nt.assert_equals(g.SRID.lat_lon_NAD83, 4269)
        nt.assert_equals(g.SRID.lat_lon_WGS84, 4326)

        nt.assert_equals(g.SRID.UPS_WGS84_north, 32661)
        nt.assert_equals(g.SRID.UPS_WGS84_south, 32761)

        nt.assert_equals(g.SRID.UTM_WGS84_north, 32600)
        nt.assert_equals(g.SRID.UTM_WGS84_south, 32700)

        nt.assert_equals(g.SRID.UTM_NAD83_northeast, 3313)
        nt.assert_equals(g.SRID.UTM_NAD83_northwest, 26900)

    def test_constructor_calls(self):
        for pt in self.pts:
            print("Constructing from point", pt)

            # Constructor from 2D point
            g.UTMUPSZone(pt)

            # Constructor from 3D point
            pt_3d = np.append(pt, 50)
            g.UTMUPSZone(pt_3d)

            # Constructor from lat lon
            g.UTMUPSZone(pt[0], pt[1])

    def test_utm_ups_zone_initial_values(self):
        z1 = g.UTMUPSZone(self.loc1)
        nt.assert_equal(z1.number, 18)
        nt.assert_equal(z1.north, True)

        z2 = g.UTMUPSZone(self.loc2)
        nt.assert_equal(z2.number, 31)
        nt.assert_equal(z2.north, True)

        z3 = g.UTMUPSZone(self.loc3)
        nt.assert_equal(z3.number, 20)
        nt.assert_equal(z3.north, True)

        z4 = g.UTMUPSZone(self.loc4)
        nt.assert_equal(z4.number, 0)
        nt.assert_equal(z4.north, True)

        z5 = g.UTMUPSZone(self.loc5)
        nt.assert_equal(z5.number, 58)
        nt.assert_equal(z5.north, False)

        z6 = g.UTMUPSZone(self.loc6)
        nt.assert_equal(z6.number, 0)
        nt.assert_equal(z6.north, False)

    def test_set_utm_ups_zone_struct_values(self):
        # Test getting and setting a few values
        z = g.UTMUPSZone(self.loc1)
        values = [(10, False), (31, True), (12, True), (0, False)]
        for num, north_bool in values:
            z.number = num
            z.north = north_bool

            nt.assert_equals(z.number, num)
            nt.assert_equals(z.north, north_bool)

    def test_bad_set_utm_ups_zone_struct_values(self):
        # Set to a valid value first
        z = g.UTMUPSZone(self.loc1)
        with nt.assert_raises(TypeError):
            z.number = "string, not int"

        with nt.assert_raises(TypeError):
            z.north = "string, not bool"

        nt.assert_equals(z.number, 18)
        nt.assert_equals(z.north, True)

    def test_zone_range_error(self):
        with nt.assert_raises(ValueError):
            g.UTMUPSZone(0, -100)

        with nt.assert_raises(ValueError):
            g.UTMUPSZone(0, 100)

    # Below functions were based off of the C++ geodesy tests
    # located in vital/tests/test_geodesy.cxx
    def get_desc_value(self, desc, key):
        return desc[key] if key in desc else "(not found)"

    def print_desc(self, name, desc):
        print(name)
        print(desc)

    def test_descriptions(self):
        modules.load_known_modules()

        # Test WGS84 lat/lon
        desc_wgs84_ll = g.geo_crs_description(g.SRID.lat_lon_WGS84)
        self.print_desc("WGS84 lat/lon", desc_wgs84_ll)

        nt.assert_equal(self.get_desc_value(desc_wgs84_ll, "datum"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_ll, "ellipse"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_ll, "projection"), "longlat")

        # Test NAD83 lat/lon
        desc_nad83_ll = g.geo_crs_description(g.SRID.lat_lon_NAD83)
        self.print_desc("NAD83 lat/lon", desc_nad83_ll)

        nt.assert_equal(self.get_desc_value(desc_nad83_ll, "datum"), "NAD83")
        nt.assert_equal(self.get_desc_value(desc_nad83_ll, "ellipse"), "GRS80")
        nt.assert_equal(self.get_desc_value(desc_nad83_ll, "projection"), "longlat")

        # Test WGS84 UTM North
        WGS84_UTM_21N = g.SRID.UTM_WGS84_north + 21
        desc_wgs84_utm_21n = g.geo_crs_description(WGS84_UTM_21N)
        self.print_desc("WGS84 UTM North 21", desc_wgs84_utm_21n)

        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_21n, "datum"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_21n, "ellipse"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_21n, "projection"), "utm")
        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_21n, "zone"), "21")
        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_21n, "hemisphere"), "north")

        # Test WGS84 UTM South
        WGS84_UTM_55S = g.SRID.UTM_WGS84_south + 55
        desc_wgs84_utm_55s = g.geo_crs_description(WGS84_UTM_55S)
        self.print_desc("WGS84 UTM South 55", desc_wgs84_utm_55s)

        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_55s, "datum"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_55s, "ellipse"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_55s, "projection"), "utm")
        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_55s, "zone"), "55")
        nt.assert_equal(self.get_desc_value(desc_wgs84_utm_55s, "hemisphere"), "south")

        # Test NAD83 UTM West
        NAD83_UTM_18S = g.SRID.UTM_NAD83_northwest + 18
        desc_nad83_utm_18s = g.geo_crs_description(NAD83_UTM_18S)
        self.print_desc("NAD83 UTM North 18", desc_nad83_utm_18s)

        nt.assert_equal(self.get_desc_value(desc_nad83_utm_18s, "datum"), "NAD83")
        nt.assert_equal(self.get_desc_value(desc_nad83_utm_18s, "ellipse"), "GRS80")
        nt.assert_equal(self.get_desc_value(desc_nad83_utm_18s, "projection"), "utm")
        nt.assert_equal(self.get_desc_value(desc_nad83_utm_18s, "zone"), "18")
        nt.assert_equal(self.get_desc_value(desc_nad83_utm_18s, "hemisphere"), "north")

        # Test NAD83 UTM East
        NAD83_UTM_59S = g.SRID.UTM_NAD83_northeast + 59
        desc_nad83_utm_59s = g.geo_crs_description(NAD83_UTM_59S)
        self.print_desc("NAD83 UTM North 59", desc_nad83_utm_59s)

        nt.assert_equal(self.get_desc_value(desc_nad83_utm_59s, "datum"), "NAD83")
        nt.assert_equal(self.get_desc_value(desc_nad83_utm_59s, "ellipse"), "GRS80")
        nt.assert_equal(self.get_desc_value(desc_nad83_utm_59s, "projection"), "utm")
        nt.assert_equal(self.get_desc_value(desc_nad83_utm_59s, "zone"), "59")
        nt.assert_equal(self.get_desc_value(desc_nad83_utm_59s, "hemisphere"), "north")

        # Test WGS84 UPS North
        desc_wgs84_ups_n = g.geo_crs_description(g.SRID.UPS_WGS84_north)
        self.print_desc("WGS84 UPS North", desc_wgs84_ups_n)

        nt.assert_equal(self.get_desc_value(desc_wgs84_ups_n, "datum"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_ups_n, "ellipse"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_ups_n, "projection"), "stere")

        # Test WGS84 UPS South
        desc_wgs84_ups_s = g.geo_crs_description(g.SRID.UPS_WGS84_south)
        self.print_desc("WGS84 UPS South", desc_wgs84_ups_s)

        nt.assert_equal(self.get_desc_value(desc_wgs84_ups_s, "datum"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_ups_s, "ellipse"), "WGS84")
        nt.assert_equal(self.get_desc_value(desc_wgs84_ups_s, "projection"), "stere")

    def test_convert(self):
        modules.load_known_modules()

        # First no altitude
        loc_ll = self.loc1
        crs_ll = g.SRID.lat_lon_WGS84
        loc_utm = np.array([601375.01, 4744863.31])
        crs_utm = g.SRID.UTM_WGS84_north + 18

        ll_to_utm = g.geo_conv(loc_ll, crs_ll, crs_utm)
        utm_to_ll = g.geo_conv(loc_utm, crs_utm, crs_ll)

        np.testing.assert_array_almost_equal(utm_to_ll, loc_ll, decimal=7)
        np.testing.assert_array_almost_equal(ll_to_utm, loc_utm, decimal=2)

        print(
            "Original in LL:", loc_ll, "Recovered after converting UTM->LL:", utm_to_ll
        )
        print(
            "Original in UTM:",
            loc_utm,
            "Recovered after converting LL->UTM:",
            ll_to_utm,
        )
