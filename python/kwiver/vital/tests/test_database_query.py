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

Tests for Python interface to vital::database_query

"""

from kwiver.vital.types import (
    database_query,
    uid,
    track_descriptor,
    Timestamp,
    GeoPolygon,
    Polygon,
    geodesy as g,
)

from kwiver.vital.tests.py_helpers import create_track_descriptor_set

import numpy as np
import unittest


class TestVitalDatabaseQuery(unittest.TestCase):
    def _create_database_query(self):
        return database_query.DatabaseQuery()

    def _create_geo_poly(self):
        p1, p2 = np.array([10, 10]), np.array([10, 50])
        return GeoPolygon(Polygon([p1, p2]), g.SRID.lat_lon_WGS84)

    def test_new(self):
        database_query.DatabaseQuery()
        database_query.DatabaseQuery()

    def test_set_and_get_id(self):
        dq = self._create_database_query()
        # First check default
        self.assertEqual(
            dq.id.value(), "", "Fresh database_query instance starts with non_empty id"
        )
        self.assertFalse(
            dq.id.is_valid(), "Fresh database_query instance starts with valid id"
        )

        # Now check setting, then getting
        test_ids = [uid.UID("first_id"), uid.UID("second_id"), uid.UID("third_id")]

        for u in test_ids:
            dq.id = u
            expected = u.value()
            actual = dq.id.value()
            self.assertEqual(
                expected,
                actual,
                "Got incorrect id. Expected {}, got {}".format(expected, actual),
            )
            self.assertTrue(
                dq.id.is_valid(),
                "database_query has invalid uid after setting to valid value",
            )

        # Try setting back to empty
        dq.id = uid.UID()
        self.assertEqual(dq.id.value(), "", "Setting id back to empty string failed")
        self.assertFalse(
            dq.id.is_valid(), "id should be invalid after setting to empty string"
        )

    def test_bad_set_id(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.id = "string, not uid"

    def _check_enum_helper(self, expected, enum_value, enum_str):
        enum_value = int(enum_value)
        self.assertEqual(
            expected,
            enum_value,
            "enum mismatch for {}. Expected {}, got {}".format(
                enum_str, expected, enum_value
            ),
        )

    def test_query_filter_enum(self):
        self._check_enum_helper(
            0, database_query.query_filter.IGNORE_FILTER, "query_filter.IGNORE_FILTER"
        )
        self._check_enum_helper(
            1,
            database_query.query_filter.CONTAINS_WHOLLY,
            "query_filter.CONTAINS_WHOLLY",
        )
        self._check_enum_helper(
            2,
            database_query.query_filter.CONTAINS_PARTLY,
            "query_filter.CONTAINS_PARTLY",
        )
        self._check_enum_helper(
            3, database_query.query_filter.INTERSECTS, "query_filter.INTERSECTS"
        )
        self._check_enum_helper(
            4,
            database_query.query_filter.INTERSECTS_INBOUND,
            "query_filter.INTERSECTS_INBOUND",
        )
        self._check_enum_helper(
            5,
            database_query.query_filter.INTERSECTS_OUTBOUND,
            "query_filter.INTERSECTS_OUTBOUND",
        )
        self._check_enum_helper(
            6,
            database_query.query_filter.DOES_NOT_CONTAIN,
            "query_filter.DOES_NOT_CONTAIN",
        )

    def test_query_type_enum(self):
        self._check_enum_helper(
            0, database_query.query_type.SIMILARITY, "query_type.SIMILARITY"
        )
        self._check_enum_helper(
            1, database_query.query_type.RETRIEVAL, "query_type.RETRIEVAL"
        )

    def test_set_and_get_type(self):
        dq = self._create_database_query()
        # First check default
        self.assertEqual(
            dq.type,
            database_query.query_type.SIMILARITY,
            "Fresh database_query instance not of query_type SIMILARITY",
        )

        # Now check setting, then getting
        test_types = [
            database_query.query_type.RETRIEVAL,
            database_query.query_type.SIMILARITY,
        ]
        for t in test_types:
            dq.type = t
            self.assertEqual(dq.type, t)

    def test_bad_set_type(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.type = "string, not query_type"

    def test_bad_set_type_outside_enum(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.type = int(database_query.query_type.RETRIEVAL) + 1

    def test_set_and_get_temporal_filter(self):
        dq = self._create_database_query()
        test_filters = [
            database_query.query_filter.IGNORE_FILTER,
            database_query.query_filter.CONTAINS_WHOLLY,
            database_query.query_filter.CONTAINS_PARTLY,
            database_query.query_filter.INTERSECTS,
            database_query.query_filter.INTERSECTS_INBOUND,
            database_query.query_filter.INTERSECTS_OUTBOUND,
            database_query.query_filter.DOES_NOT_CONTAIN,
        ]

        for f in test_filters:
            dq.temporal_filter = f
            self.assertEqual(dq.temporal_filter, f)

    def test_bad_set_temporal_filter(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.temporal_filter = "string, not query_filter"

    def test_bad_set_temporal_filter_outside_enum(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.temporal_filter = int(database_query.query_filter.DOES_NOT_CONTAIN) + 1

    def test_set_and_get_spatial_filter(self):
        dq = self._create_database_query()
        test_filters = [
            database_query.query_filter.IGNORE_FILTER,
            database_query.query_filter.CONTAINS_WHOLLY,
            database_query.query_filter.CONTAINS_PARTLY,
            database_query.query_filter.INTERSECTS,
            database_query.query_filter.INTERSECTS_INBOUND,
            database_query.query_filter.INTERSECTS_OUTBOUND,
            database_query.query_filter.DOES_NOT_CONTAIN,
        ]

        for f in test_filters:
            dq.spatial_filter = f
            self.assertEqual(dq.spatial_filter, f)

    def test_bad_set_spatial_filter_wrong_type(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.spatial_filter = "string, not query_filter"

    def test_bad_set_spatial_filter_outside_enum(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.spatial_filter = int(database_query.query_filter.DOES_NOT_CONTAIN) + 1

    def test_set_and_get_spatial_region(self):
        dq = self._create_database_query()

        # Test default
        self.assertTrue(dq.spatial_region.is_empty())

        # Try setting to non empty geo_polygon
        g_poly = self._create_geo_poly()
        dq.spatial_region = g_poly

        dq_vertices = dq.spatial_region.polygon().get_vertices()
        g_poly_vertices = g_poly.polygon().get_vertices()
        np.testing.assert_array_equal(dq_vertices, g_poly_vertices)

        # Try setting back to empty geo_polygon
        dq.spatial_region = GeoPolygon()
        self.assertTrue(dq.spatial_region.is_empty())

    def test_bad_set_spatial_region(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.spatial_region = "string, not geo_polygon"

    def test_set_and_get_stream_filter(self):
        dq = self._create_database_query()
        self.assertEqual(dq.stream_filter, "", "Default stream filter not empty string")

        # Now test setting and getting some values
        test_strs = [
            "first_stream_filter",
            "second_stream_filter",
            "",
            "fourth_stream_filter",
        ]
        for s in test_strs:
            dq.stream_filter = s
            self.assertEqual(
                dq.stream_filter, s, "Setting stream_filter to {} failed".format(s)
            )

    def test_bad_set_stream_filter(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.stream_filter = 10

    def test_set_and_get_descriptors(self):
        dq = self._create_database_query()

        self.assertEqual(dq.descriptors, [])

        # Test getting and setting a few values
        (td_set, lists_used) = create_track_descriptor_set()
        dq.descriptors = td_set
        for td, l in zip(dq.descriptors, lists_used):
            np.testing.assert_array_almost_equal(td.get_descriptor().todoublearray(), l)

        # Modifying an element of the list reflects in dq.descriptors
        td_set[0][0] += 10
        self.assertAlmostEqual(dq.descriptors[0][0], td_set[0][0])

        # But changing the list itself shouldn't
        self.assertEqual(len(dq.descriptors), len(td_set))

        new_td = track_descriptor.TrackDescriptor.create("new_td")
        new_td.resize_descriptor(3, 10)

        td_set.append(new_td)
        self.assertNotEqual(len(dq.descriptors), len(td_set))

        dq.descriptors = []
        self.assertEqual(dq.descriptors, [])

    def test_bad_set_descriptors(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.descriptors = "string, not track_descriptor_set"

    def test_set_and_get_threshold(self):
        dq = self._create_database_query()
        # First check default
        self.assertAlmostEqual(dq.threshold, 0.0)

        # Now check setting and getting some values
        test_thresholds = [4.0, 3.14, 0.0, 10, 2.71]
        for t in test_thresholds:
            dq.threshold = t
            self.assertAlmostEqual(dq.threshold, t)

    def test_bad_set_threshold(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.threshold = "string, not double"

    def test_set_and_get_temporal_bounds(self):
        dq = self._create_database_query()
        # First check the defaults
        self.assertFalse(dq.temporal_lower_bound().is_valid())
        self.assertFalse(dq.temporal_upper_bound().is_valid())

        test_bounds = [
            (Timestamp(100, 1), Timestamp(100, 1)),
            (Timestamp(100, 1), Timestamp(200, 2)),
            (Timestamp(300, 5), Timestamp(400, 6)),
        ]

        for t1, t2 in test_bounds:
            dq.set_temporal_bounds(t1, t2)
            self.assertEqual(dq.temporal_lower_bound(), t1)
            self.assertEqual(dq.temporal_upper_bound(), t2)

        # Set to default constructed timestamps
        dq.set_temporal_bounds(Timestamp(), Timestamp())
        self.assertFalse(dq.temporal_lower_bound().is_valid())
        self.assertFalse(dq.temporal_upper_bound().is_valid())

    def test_bad_set_bounds_logic_error(self):
        with self.assertRaises(RuntimeError):
            dq = self._create_database_query()
            dq.set_temporal_bounds(Timestamp(200, 2), Timestamp(100, 1))

    def test_bad_set_bounds(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.set_temporal_bounds("string", "another_string")

    def test_empty_set_bounds(self):
        with self.assertRaises(TypeError):
            dq = self._create_database_query()
            dq.set_temporal_bounds()
