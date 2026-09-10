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

Tests for vital::query_result interface

"""

import numpy as np

from kwiver.vital.types import (
    BoundingBoxD as BoundingBox,
    DetectedObject,
    DetectedObjectType,
    geodesy as g,
    GeoPoint,
    Image,
    ImageContainer,
    ObjectTrackSet,
    ObjectTrackState,
    QueryResult,
    Timestamp,
    Track,
    track_descriptor,
    UID,
)
from kwiver.vital.tests.py_helpers import create_track_descriptor_set
import unittest


class TestVitalQueryResult(unittest.TestCase):
    def test_create(self):
        QueryResult()

    def _create_query_result(self):
        return QueryResult()

    # Helper function to create an object_track_set
    # See TestObjectTrackSet._create_track
    def _create_object_track_set(self):
        bbox = BoundingBox(10, 10, 20, 20)
        dot = DetectedObjectType("test", 0.4)
        do = DetectedObject(bbox, 0.4, dot)
        track = Track()
        for i in range(10):
            track.append(ObjectTrackState(i, i, do))
        return ObjectTrackSet([track])

    def test_set_and_get_query_id(self):
        qr = self._create_query_result()

        # First check default
        self.assertEqual(qr.query_id.value(), "")
        self.assertFalse(qr.query_id.is_valid())

        # Now check setting and getting a few values
        qr.query_id = UID("first")
        self.assertEqual(qr.query_id.value(), "first")

        qr.query_id = UID("second")
        self.assertEqual(qr.query_id.value(), "second")

        qr.query_id = UID("42")
        self.assertEqual(qr.query_id.value(), "42")

        # Try setting back to empty
        qr.query_id = UID()
        self.assertEqual(qr.query_id.value(), "")

    def test_bad_set_query_id(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.query_id = "string, not uid"

    def test_set_and_get_stream_id(self):
        qr = self._create_query_result()

        self.assertEqual(qr.stream_id, "")

        qr.stream_id = "first"
        self.assertEqual(qr.stream_id, "first")

        qr.stream_id = "second"
        self.assertEqual(qr.stream_id, "second")

        qr.stream_id = "42"
        self.assertEqual(qr.stream_id, "42")

        qr.stream_id = ""
        self.assertEqual(qr.stream_id, "")

    def test_bad_set_stream_id(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.stream_id = 5

    def test_set_and_get_instance_id(self):
        qr = self._create_query_result()

        qr.instance_id = 1
        self.assertEqual(qr.instance_id, 1)

        qr.instance_id += 1
        self.assertEqual(qr.instance_id, 2)

        qr.instance_id = 1234
        self.assertEqual(qr.instance_id, 1234)

        qr.instance_id = 0
        self.assertEqual(qr.instance_id, 0)

    def test_bad_set_instance_id(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.instance_id = "string, not numeric"

    def test_set_and_get_relevancy_score(self):
        qr = self._create_query_result()

        qr.relevancy_score = 1
        self.assertEqual(qr.relevancy_score, 1)

        d = 1234567
        qr.relevancy_score = d
        self.assertEqual(qr.relevancy_score, d)

        # Check precision
        d = 1234.5678901234567
        qr.relevancy_score = d
        np.testing.assert_almost_equal(qr.relevancy_score, d, decimal=15)

        qr.relevancy_score = -d
        np.testing.assert_almost_equal(qr.relevancy_score, -d, decimal=15)

        qr.relevancy_score = 0
        self.assertEqual(qr.relevancy_score, 0)

    def test_bad_set_relevancy_score(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.relevancy_score = "string, not double"

    def test_set_and_get_temporal_bounds(self):
        qr = self._create_query_result()

        # First check the defaults
        self.assertFalse(qr.start_time().is_valid())
        self.assertFalse(qr.end_time().is_valid())

        test_bounds = [
            (Timestamp(100, 1), Timestamp(100, 1)),
            (Timestamp(100, 1), Timestamp(200, 2)),
            (Timestamp(300, 5), Timestamp(400, 6)),
        ]

        for t1, t2 in test_bounds:
            qr.set_temporal_bounds(t1, t2)
            self.assertEqual(qr.start_time(), t1)
            self.assertEqual(qr.end_time(), t2)

        qr.set_temporal_bounds(Timestamp(), Timestamp())
        self.assertFalse(qr.start_time().is_valid())
        self.assertFalse(qr.end_time().is_valid())

    def test_bad_set_bounds(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.set_temporal_bounds("string", "another_string")

    def test_set_and_get_location(self):
        qr = self._create_query_result()
        crs_ll = g.SRID.lat_lon_WGS84
        crs_utm_18n = g.SRID.UTM_WGS84_north + 18

        self.assertTrue(qr.location.is_empty())

        p = GeoPoint(np.array([-73.759291, 42.849631]), crs_ll)
        qr.location = p
        self.assertEqual(qr.location.crs(), crs_ll)
        np.testing.assert_array_almost_equal(qr.location.location(), p.location())

        # Increase the precision, add altitude
        p = GeoPoint(np.array([77.397577193572642, 38.17996907564275, 50]), crs_ll)
        qr.location = p
        self.assertEqual(qr.location.crs(), crs_ll)
        np.testing.assert_array_almost_equal(
            qr.location.location(), p.location(), decimal=15
        )

        # Try different crs
        p = GeoPoint(np.array([601375.01, 4744863.31]), crs_utm_18n)
        qr.location = p
        self.assertEqual(qr.location.crs(), crs_utm_18n)
        np.testing.assert_array_almost_equal(qr.location.location(), p.location())

        # Back to empty
        qr.location = GeoPoint()
        self.assertTrue(qr.location.is_empty())

    def test_bad_set_location(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.location = 5

    def test_set_and_get_tracks(self):
        qr = self._create_query_result()

        ots = self._create_object_track_set()
        qr.tracks = ots
        self.assertEqual(qr.tracks.size(), ots.size())
        self.assertEqual(ots.size(), 1)

        # Changes made to the elements of object_track_set should reflect
        # in qr.tracks, since they are pointers
        # Establish id of 0 in both
        self.assertEqual(qr.tracks.tracks()[0].id, ots.tracks()[0].id)
        self.assertEqual(ots.tracks()[0].id, 0)

        # Change one of them
        ots.tracks()[0].id = 1

        # See changes in both
        self.assertEqual(qr.tracks.tracks()[0].id, ots.tracks()[0].id)
        self.assertEqual(ots.tracks()[0].id, 1)

        # Set to default-constructed object
        qr.tracks = ObjectTrackSet()
        self.assertEqual(qr.tracks.size(), 0)

    def test_bad_set_tracks(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.tracks = "string, not object_track_set"

    def test_set_and_get_descriptors(self):
        qr = self._create_query_result()

        self.assertEqual(qr.descriptors, [])

        # Test getting and setting a few values
        (td_set, lists_used) = create_track_descriptor_set()
        qr.descriptors = td_set
        for td, l in zip(qr.descriptors, lists_used):
            np.testing.assert_array_almost_equal(td.get_descriptor().todoublearray(), l)

        # Modifying an element of the list reflects in qr.descriptors
        td_set[0][0] += 10
        self.assertAlmostEqual(qr.descriptors[0][0], td_set[0][0])

        # But changing the list itself shouldn't
        self.assertEqual(len(qr.descriptors), len(td_set))

        new_td = track_descriptor.TrackDescriptor.create("new_td")
        new_td.resize_descriptor(3, 10)

        td_set.append(new_td)
        self.assertNotEqual(len(qr.descriptors), len(td_set))

        qr.descriptors = []
        self.assertEqual(qr.descriptors, [])

    def test_bad_set_descriptors(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.descriptors = "string, not track_descriptor_set"

    def test_set_and_get_image_data(self):
        qr = self._create_query_result()

        imc_list = [ImageContainer(Image())]
        qr.image_data = imc_list
        self.assertEqual(len(qr.image_data), len(imc_list))
        self.assertEqual(len(imc_list), 1)
        self.assertEqual(qr.image_data[0].size(), imc_list[0].size())
        self.assertEqual(imc_list[0].size(), 0)

        imc_list.append(ImageContainer(Image(720, 480)))
        qr.image_data = imc_list
        self.assertEqual(len(qr.image_data), len(imc_list))
        self.assertEqual(len(imc_list), 2)
        self.assertEqual(qr.image_data[0].size(), imc_list[0].size())
        self.assertEqual(imc_list[0].size(), 0)
        self.assertEqual(qr.image_data[1].size(), imc_list[1].size())
        self.assertEqual(imc_list[1].size(), 720 * 480)

        qr.image_data = []
        self.assertEqual(len(qr.image_data), 0)

    def test_bad_set_image_data(self):
        with self.assertRaises(TypeError):
            qr = self._create_query_result()
            qr.image_data = "string, not image_data"
