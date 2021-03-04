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

# Tests for Python interface to vital::geo_polygon

# """

import nose.tools as nt
import numpy as np

from kwiver.vital.types import geo_polygon as gp, geodesy, Polygon
from kwiver.vital.config import config
from kwiver.vital.modules import modules


class TestVitalGeoPolygon(object):
    # Creates a regular (not Geo) polygon for a few tests
    def _create_polygon(self):
        return Polygon(
            [
                np.array([10, 10]),
                np.array([10, 50]),
                np.array([50, 50]),
                np.array([30, 30])
            ]
        )

    def setUp(self):
        self.loc_ll = np.array([-149.484444, -17.619482])
        self.loc_utm = np.array([236363.98, 8050181.74])

        self.loc2_ll = np.array([-73.759291, 42.849631])

        self.crs_ll = geodesy.SRID.lat_lon_WGS84
        self.crs_utm_6s = geodesy.SRID.UTM_WGS84_south + 6

    def test_create(self):
        gp.GeoPolygon()
        gp.GeoPolygon(self._create_polygon(), self.crs_ll)

    def test_inital_is_empty(self):
        g_poly1 = gp.GeoPolygon()
        g_poly2 = gp.GeoPolygon(self._create_polygon(), self.crs_ll)
        nt.ok_(g_poly1.is_empty())
        nt.assert_false(g_poly2.is_empty())

    def test_initial_crs(self):
        g_poly1 = gp.GeoPolygon()
        g_poly2 = gp.GeoPolygon(self._create_polygon(), self.crs_ll)
        nt.assert_equals(g_poly1.crs(), -1)
        nt.assert_equals(g_poly2.crs(), self.crs_ll)


    def test_initial_polygon(self):
        g_poly = gp.GeoPolygon(self._create_polygon(), self.crs_ll)
        poly = self._create_polygon()

        np.testing.assert_array_equal(
            g_poly.polygon().get_vertices(), poly.get_vertices()
        )
        np.testing.assert_array_equal(
            g_poly.polygon(self.crs_ll).get_vertices(), poly.get_vertices()
        )

    def test_no_polygon_for_key(self):
        g_poly = gp.GeoPolygon()
        nt.assert_raises(IndexError, g_poly.polygon)
        nt.assert_raises(IndexError, g_poly.polygon, self.crs_ll)

    # Try setting the polygon member by using the above locations.
    def test_set_polygon(self):
        g_poly1 = gp.GeoPolygon()
        g_poly2 = gp.GeoPolygon(self._create_polygon(), self.crs_ll)
        modules.load_known_modules()
        for instance in [g_poly1, g_poly2]:

            instance.set_polygon(Polygon([self.loc2_ll]), self.crs_ll)

            nt.assert_equals(instance.polygon().num_vertices(), 1)
            nt.assert_equals(instance.crs(), self.crs_ll)
            np.testing.assert_array_almost_equal(instance.polygon().at(0), self.loc2_ll)
            np.testing.assert_array_almost_equal(
                instance.polygon(self.crs_ll).at(0), self.loc2_ll
            )
            nt.assert_false(instance.is_empty())
            instance.set_polygon(Polygon([self.loc_utm]), self.crs_utm_6s)

            nt.assert_equals(instance.polygon().num_vertices(), 1)
            nt.assert_equals(instance.crs(), self.crs_utm_6s)
            np.testing.assert_array_almost_equal(instance.polygon().at(0), self.loc_utm)
            np.testing.assert_array_almost_equal(
                instance.polygon(self.crs_utm_6s).at(0), self.loc_utm
            )
            nt.assert_false(instance.is_empty())

            try:
                loc_out = instance.polygon(self.crs_ll).at(0)
            except:
                # This may throw an exception regarding no conversion functor.
                # This is ok. See test_geo_polygon.cxx
                pass
            else:
                diff_loc_out_loc2_ll = np.linalg.norm(loc_out - self.loc2_ll)
                nt.assert_not_equal(
                    diff_loc_out_loc2_ll,
                    0,
                    msg="Changing the location did not clear the location cache",
                ) # This would fail if loc2_ll was cached

    def test_conversion(self):
        modules.load_known_modules()

        p_ll = gp.GeoPolygon(Polygon([self.loc_ll]), self.crs_ll)
        p_utm = gp.GeoPolygon(Polygon([self.loc_utm]), self.crs_utm_6s)

        conv_loc_utm = p_ll.polygon(p_utm.crs()).at(0)
        conv_loc_ll = p_utm.polygon(p_ll.crs()).at(0)

        np.testing.assert_array_almost_equal(
            p_ll.polygon().at(0), conv_loc_ll, decimal=7
        )
        np.testing.assert_array_almost_equal(
            p_utm.polygon().at(0), conv_loc_utm, decimal=2
        )

        np.testing.assert_array_almost_equal(self.loc_ll, conv_loc_ll, decimal=7)
        np.testing.assert_array_almost_equal(self.loc_utm, conv_loc_utm, decimal=2)

    def test_to_str_empty(self):
        g_poly1 = gp.GeoPolygon()
        nt.assert_equals(str(g_poly1), "{ empty }")
        print("empty geo_polygon:", str(g_poly1), sep='\n')

    # Also test double roundtrip
    def test_to_str(self):
        x_in = 12.3456789012345678
        y_in = -12.3456789012345678
        input_array = [np.array([x_in, y_in])]
        g_poly = gp.GeoPolygon(Polygon(input_array), self.crs_ll)

        split_str = str(g_poly).split()

        # Initial character
        nt.assert_equals(split_str[0], "{")

        # x coord
        np.testing.assert_almost_equal(float(split_str[1]), x_in, decimal=15)

        # Separator
        nt.assert_equals(split_str[2], "/")

        # y coord
        np.testing.assert_almost_equal(float(split_str[3]), y_in, decimal=15)

        # Closing character
        nt.assert_equals(split_str[4], "}")

        # @
        nt.assert_equals(split_str[5], "@")

        # crs_ll
        nt.assert_equals(int(split_str[6]), self.crs_ll)

        print("geo_polygon with data:", str(g_poly), sep='\n')
