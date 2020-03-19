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

# Tests for Python interface to vital::geo_point

# """

import nose.tools as nt
import numpy as np

from kwiver.vital.types import geo_point as gp, geodesy
from kwiver.vital.modules import modules


class TestVitalGeoPoint(object):
    def setUp(self):
        self.loc1 = np.array([-73.759291, 42.849631])
        self.loc2 = np.array([-73.757161, 42.849764])
        self.loc3 = np.array([601375.01, 4744863.31])

        # Points with altitude
        self.loc1a = np.array([-73.759291, 42.849631, 50])
        self.loc2a = np.array([-73.757161, 42.849764, 50])
        self.loc3a = np.array([601375.01, 4744863.31, 50])

        self.crs_ll = geodesy.SRID.lat_lon_WGS84
        self.crs_utm_18n = geodesy.SRID.UTM_WGS84_north + 18

    def _create_points(self):
        return (
            gp.GeoPoint(),
            gp.GeoPoint(self.loc1, self.crs_ll),
            gp.GeoPoint(self.loc1a, self.crs_ll),
        )

    def test_create(self):
        gp.GeoPoint()
        gp.GeoPoint(self.loc1, self.crs_ll)
        gp.GeoPoint(self.loc1a, self.crs_ll)

    def test_initial_is_empty(self):
        p1, p2, p3 = self._create_points()
        nt.ok_(p1.is_empty())
        nt.assert_false(p2.is_empty())
        nt.assert_false(p3.is_empty())

    def test_initial_crs(self):
        p1, p2, p3 = self._create_points()
        nt.assert_equals(p1.crs(), -1)
        nt.assert_equals(p2.crs(), self.crs_ll)
        nt.assert_equals(p3.crs(), self.crs_ll)

    def test_initial_location(self):
        _, p2, p3 = self._create_points()

        # p2
        loc_expected = np.concatenate([self.loc1, [0]])
        np.testing.assert_array_almost_equal(p2.location(), loc_expected)
        np.testing.assert_array_almost_equal(p2.location(self.crs_ll), loc_expected)

        # p3
        np.testing.assert_array_almost_equal(p3.location(), self.loc1a)
        np.testing.assert_array_almost_equal(p3.location(self.crs_ll), self.loc1a)

    def test_no_location_for_key(self):
        p1 = gp.GeoPoint()
        nt.assert_raises(IndexError, p1.location)
        nt.assert_raises(IndexError, p1.location, self.crs_ll)

    def test_set_location(self):
        _, p2, _ = self._create_points()

        # Change location and crs
        p2.set_location(self.loc3, self.crs_utm_18n)
        loc3_expected = np.concatenate([self.loc3, [0]])

        nt.assert_false(p2.is_empty())
        nt.assert_equals(p2.crs(), self.crs_utm_18n)
        np.testing.assert_array_almost_equal(p2.location(), loc3_expected)
        np.testing.assert_array_almost_equal(
            p2.location(self.crs_utm_18n), loc3_expected
        )

        # Change values again
        p2.set_location(self.loc2, self.crs_ll)
        loc2_expected = np.concatenate([self.loc2, [0]])

        nt.assert_false(p2.is_empty())
        nt.assert_equals(p2.crs(), self.crs_ll)
        np.testing.assert_array_almost_equal(p2.location(), loc2_expected)
        np.testing.assert_array_almost_equal(p2.location(self.crs_ll), loc2_expected)

        # Test location isn't cached
        try:
            loc2_utm = p2.location(self.crs_utm_18n)
        except:
            # This may throw an exception regarding no conversion functor.
            # This is ok. See test_geo_point.cxx
            pass

        else:
            diff_loc2_loc3_utm = np.linalg.norm(loc2_utm - loc3_expected)
            nt.assert_not_equal(
                diff_loc2_loc3_utm,
                0,
                msg="Changing the location did not clear the location cache",
            )  # This would fail if loc3 was cached

    def test_conversion(self):
        modules.load_known_modules()

        p_ll = gp.GeoPoint(self.loc1, self.crs_ll)
        p_utm = gp.GeoPoint(self.loc3, self.crs_utm_18n)

        conv_loc_utm = p_ll.location(p_utm.crs())
        conv_loc_ll = p_utm.location(p_ll.crs())

        loc3_expected = np.concatenate([self.loc3, [0]])
        epsilon_ll_to_utm = np.linalg.norm(loc3_expected - conv_loc_utm)

        loc1_expected = np.concatenate([self.loc1, [0]])
        epsilon_utm_to_ll = np.linalg.norm(loc1_expected - conv_loc_ll)

        np.testing.assert_array_almost_equal(p_ll.location(), conv_loc_ll, decimal=7)
        np.testing.assert_array_almost_equal(p_utm.location(), conv_loc_utm, decimal=2)
        nt.ok_(epsilon_ll_to_utm < 10**(-2))
        nt.ok_(epsilon_utm_to_ll < 10**(-7))

        print("LL->UTM epsilon:", epsilon_ll_to_utm)
        print("UTM->LL epsilon:", epsilon_utm_to_ll)

        # Tests with altitude
        p_lla = gp.GeoPoint(self.loc1a, self.crs_ll)
        p_utma = gp.GeoPoint(self.loc3a, self.crs_utm_18n)

        conv_loc_utma = p_lla.location(p_utma.crs())
        conv_loc_lla = p_utma.location(p_lla.crs())

        epsilon_lla_to_utma = np.linalg.norm(self.loc3a - conv_loc_utma)
        epsilon_utma_to_lla = np.linalg.norm(self.loc1a - conv_loc_lla)

        np.testing.assert_array_almost_equal(p_lla.location(), conv_loc_lla, decimal=7)
        np.testing.assert_array_almost_equal(
            p_utma.location(), conv_loc_utma, decimal=2
        )
        nt.ok_(epsilon_lla_to_utma < 10**(-2))
        nt.ok_(epsilon_utma_to_lla < 10**(-7))

        print("LLa->UTMa epsilon:", epsilon_lla_to_utma)
        print("UTMa->LLa epsilon:", epsilon_utma_to_lla)

    def test_to_str_empty(self):
        p1 = gp.GeoPoint()
        nt.assert_equals(str(p1), "geo_point\n[ empty ]")
        print("empty geo_point to string:", str(p1), sep='\n')

    # Also make sure the doubles roundtrip
    def test_to_str(self):
        easting_in = 12.3456789012345678
        northing_in = -12.3456789012345678
        p1 = gp.GeoPoint(np.array([easting_in, northing_in]), self.crs_ll)

        split_str = str(p1).split()

        # Initial strings
        nt.assert_equals(split_str[0], "geo_point")
        nt.assert_equals(split_str[1], "[")

        # Parsing easting
        easting_out, comma = split_str[2][:-1], split_str[2][-1]
        nt.assert_equals(comma, ",")
        np.testing.assert_almost_equal(float(easting_out), easting_in, decimal=15)

        # Parsing northing
        northing_out, comma = split_str[3][:-1], split_str[3][-1]
        nt.assert_equals(comma, ",")
        np.testing.assert_almost_equal(float(northing_out), northing_in, decimal=15)

        # Altitude (0)
        altitude_out = split_str[4]
        nt.assert_almost_equal(float(altitude_out), 0)

        nt.assert_equals(split_str[5], "]")
        nt.assert_equals(split_str[6], "@")
        nt.assert_equals(int(split_str[7]), self.crs_ll)

        print("geo_point with data:", str(p1), sep='\n')
