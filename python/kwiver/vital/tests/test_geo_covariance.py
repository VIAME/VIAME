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

# Tests for Python interface to vital::geo_covariance

# """

import nose.tools as nt
import numpy as np

from kwiver.vital.types import Covar3f, GeoCovariance, geodesy, GeoPoint
from kwiver.vital.modules import modules


class TestVitalGeoCovariance(object):
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

    def _create_geo_covar(self):
        return (
            GeoCovariance(),
            GeoCovariance(self.loc1, self.crs_ll),
            GeoCovariance(self.loc1a, self.crs_ll),
        )

    def test_create(self):
        GeoCovariance()
        GeoCovariance(self.loc1, self.crs_ll)
        GeoCovariance(self.loc1a, self.crs_ll)

    def test_set_and_get_covariance(self):
        gcs = self._create_geo_covar()

        for gc in gcs:
            c = Covar3f()
            gc.covariance = c
            mat_in = c.matrix()
            mat_out = gc.covariance.matrix()
            np.testing.assert_array_equal(mat_in, mat_out)

            c = Covar3f(-5.5)
            gc.covariance = c
            mat_in = c.matrix()
            mat_out = gc.covariance.matrix()
            np.testing.assert_array_almost_equal(mat_in, mat_out)

            # Generate 3x3 matrix of floats between -5 and 5
            mat_in = np.ndarray((3, 3))
            mat_in[:] = 5
            c = Covar3f(mat_in)
            gc.covariance = c
            mat_out = gc.covariance.matrix()
            np.testing.assert_array_almost_equal(mat_in, c.matrix())

    def test_inheritance(self):
        gcs = self._create_geo_covar()
        for gc in gcs:
            nt.ok_(isinstance(gc, GeoPoint))

    def test_initial_is_empty(self):
        gc1, gc2, gc3 = self._create_geo_covar()
        nt.ok_(gc1.is_empty())
        nt.assert_false(gc2.is_empty())
        nt.assert_false(gc3.is_empty())

    def test_initial_crs(self):
        gc1, gc2, gc3 = self._create_geo_covar()
        nt.assert_equals(gc1.crs(), -1)
        nt.assert_equals(gc2.crs(), self.crs_ll)
        nt.assert_equals(gc3.crs(), self.crs_ll)

    def test_initial_location(self):
        _, gc2, gc3 = self._create_geo_covar()

        # gc2
        loc_expected = np.concatenate([self.loc1, [0]])
        np.testing.assert_array_almost_equal(gc2.location(), loc_expected)
        np.testing.assert_array_almost_equal(gc2.location(self.crs_ll), loc_expected)

        # gc3
        np.testing.assert_array_almost_equal(gc3.location(), self.loc1a)
        np.testing.assert_array_almost_equal(gc3.location(self.crs_ll), self.loc1a)

    def test_no_location_for_key(self):
        gc1 = GeoCovariance()
        nt.assert_raises(IndexError, gc1.location)
        nt.assert_raises(IndexError, gc1.location, self.crs_ll)

    def test_set_location(self):
        _, gc2, _ = self._create_geo_covar()

        # Change location and crs
        gc2.set_location(self.loc3, self.crs_utm_18n)
        loc3_expected = np.concatenate([self.loc3, [0]])

        nt.assert_false(gc2.is_empty())
        nt.assert_equals(gc2.crs(), self.crs_utm_18n)
        np.testing.assert_array_almost_equal(gc2.location(), loc3_expected)
        np.testing.assert_array_almost_equal(
            gc2.location(self.crs_utm_18n), loc3_expected
        )

        # Change values again
        gc2.set_location(self.loc2, self.crs_ll)
        loc2_expected = np.concatenate([self.loc2, [0]])

        nt.assert_false(gc2.is_empty())
        nt.assert_equals(gc2.crs(), self.crs_ll)
        np.testing.assert_array_almost_equal(gc2.location(), loc2_expected)
        np.testing.assert_array_almost_equal(gc2.location(self.crs_ll), loc2_expected)

        # Test location isn't cached
        try:
            loc2_utm = gc2.location(self.crs_utm_18n)
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

        gc_ll = GeoCovariance(self.loc1, self.crs_ll)
        gc_utm = GeoCovariance(self.loc3, self.crs_utm_18n)

        conv_loc_utm = gc_ll.location(gc_utm.crs())
        conv_loc_ll = gc_utm.location(gc_ll.crs())

        loc3_expected = np.concatenate([self.loc3, [0]])
        epsilon_ll_to_utm = np.linalg.norm(loc3_expected - conv_loc_utm)

        loc1_expected = np.concatenate([self.loc1, [0]])
        epsilon_utm_to_ll = np.linalg.norm(loc1_expected - conv_loc_ll)

        np.testing.assert_array_almost_equal(gc_ll.location(), conv_loc_ll, decimal=7)
        np.testing.assert_array_almost_equal(gc_utm.location(), conv_loc_utm, decimal=2)
        nt.ok_(epsilon_ll_to_utm < 10 ** (-2))
        nt.ok_(epsilon_utm_to_ll < 10 ** (-7))

        print("LL->UTM epsilon:", epsilon_ll_to_utm)
        print("UTM->LL epsilon:", epsilon_utm_to_ll)

        # Tests with altitude
        gc_lla = GeoCovariance(self.loc1a, self.crs_ll)
        gc_utma = GeoCovariance(self.loc3a, self.crs_utm_18n)

        conv_loc_utma = gc_lla.location(gc_utma.crs())
        conv_loc_lla = gc_utma.location(gc_lla.crs())

        epsilon_lla_to_utma = np.linalg.norm(self.loc3a - conv_loc_utma)
        epsilon_utma_to_lla = np.linalg.norm(self.loc1a - conv_loc_lla)

        np.testing.assert_array_almost_equal(gc_lla.location(), conv_loc_lla, decimal=7)
        np.testing.assert_array_almost_equal(
            gc_utma.location(), conv_loc_utma, decimal=2
        )
        nt.ok_(epsilon_lla_to_utma < 10 ** (-2))
        nt.ok_(epsilon_utma_to_lla < 10 ** (-7))

        print("LLa->UTMa epsilon:", epsilon_lla_to_utma)
        print("UTMa->LLa epsilon:", epsilon_utma_to_lla)

    def test_to_str_empty(self):
        gc = GeoCovariance()
        nt.assert_equals(str(gc), "geo_covariance\n[ empty ]")
        print("empty geo_covariance to string:", str(gc), sep="\n")

    def test_to_str(self):
        easting_in = 12.3456789012345678
        northing_in = -12.3456789012345678
        gc = GeoCovariance(np.array([easting_in, northing_in]), self.crs_ll)

        split_str = str(gc).split()

        # Initial strings
        nt.assert_equals(split_str[0], "geo_covariance")
        nt.assert_equals(split_str[1], "-")
        nt.assert_equals(split_str[2], "value")
        nt.assert_equals(split_str[3], ":")
        nt.assert_equals(split_str[4], "[")

        # Parsing easting
        easting_out, comma = split_str[5][:-1], split_str[5][-1]
        nt.assert_equals(comma, ",")
        np.testing.assert_almost_equal(float(easting_out), easting_in, decimal=15)

        # Parsing northing
        northing_out, comma = split_str[6][:-1], split_str[6][-1]
        nt.assert_equals(comma, ",")
        np.testing.assert_almost_equal(float(northing_out), northing_in, decimal=15)

        # Altitude (0)
        altitude_out = split_str[7]
        nt.assert_almost_equal(float(altitude_out), 0)

        nt.assert_equals(split_str[8], "]")
        nt.assert_equals(split_str[9], "@")
        nt.assert_equals(int(split_str[10]), self.crs_ll)

        print("geo_point with data:", str(gc), sep="\n")
