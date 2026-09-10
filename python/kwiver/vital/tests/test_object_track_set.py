"""
ckwg +31
Copyright 2018-2020 by Kitware, Inc.
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

Tests for ObjectTrackSet

"""

import unittest

import numpy

from kwiver.vital.types import (
    ObjectTrackSet,
    ObjectTrackState,
    BoundingBoxD as bbD,
    DetectedObjectType as DOT,
    DetectedObject,
    Track,
)


class TestObjectTrackSet(unittest.TestCase):
    def _create_track(self):
        """
        Helper function to create a track
        :return: Track with 10 object track state. Every track state has same
                    detected object however the fram number and time varies from
                    [0, 10)
        """
        bbox = bbD(10, 10, 20, 20)
        cm = DOT("test", 0.4)
        do = DetectedObject(bbox, 0.4, cm)
        track = Track()
        for i in range(10):
            track.append(ObjectTrackState(i, i, do))
        return track

    def test_new_ts(self):
        """
        Test creation of object track set
        """
        track = self._create_track()
        ObjectTrackSet([track])

    def test_all_frame_ids(self):
        """
        Test all frame ids stored in object track set are between [0, 10)
        """
        obs = ObjectTrackSet([self._create_track()])
        self.assertEqual(obs.all_frame_ids(), set(range(10)))

    def test_first_frame(self):
        """
        Test first frame id in the track set is 0
        """
        obs = ObjectTrackSet([self._create_track()])
        self.assertEqual(obs.first_frame(), 0)

    def test_last_frame(self):
        """
        Test last frame id in the track set is 9
        """
        obs = ObjectTrackSet([self._create_track()])
        self.assertEqual(obs.last_frame(), 9)

    def test_tracks(self):
        """
        Test indexed retrival of tracks.
        """
        obj_track = self._create_track()
        obs = ObjectTrackSet([obj_track])
        self.assertEqual(obs.tracks()[0], obj_track)

    def test_get_track(self):
        """
        Test track retrival using track id
        """
        obj_track = self._create_track()
        obs = ObjectTrackSet([obj_track])
        self.assertEqual(obs.get_track(0), obj_track)

    def test_track_attributes_in_object_track_set(self):
        """
        Test that track attributes work when track is in ObjectTrackSet
        """
        track = self._create_track()

        # Set attributes on the track before adding to set
        track.set_attribute("species", "fish")
        track.set_attribute("length_cm", 42.5)
        track.set_attribute("is_verified", True)
        track.set_attribute("count", 10)

        obs = ObjectTrackSet([track])

        # Retrieve the track from the set and verify attributes
        retrieved_track = obs.get_track(0)

        self.assertTrue(retrieved_track.has_attribute("species"))
        self.assertEqual(retrieved_track.get_attribute("species"), "fish")

        self.assertTrue(retrieved_track.has_attribute("length_cm"))
        self.assertAlmostEqual(retrieved_track.get_attribute("length_cm"), 42.5)

        self.assertTrue(retrieved_track.has_attribute("is_verified"))
        self.assertEqual(retrieved_track.get_attribute("is_verified"), True)

        self.assertTrue(retrieved_track.has_attribute("count"))
        self.assertEqual(retrieved_track.get_attribute("count"), 10)

        # Test attribute_keys
        keys = retrieved_track.attribute_keys()
        self.assertEqual(len(keys), 4)
        self.assertIn("species", keys)
        self.assertIn("length_cm", keys)
        self.assertIn("is_verified", keys)
        self.assertIn("count", keys)

    def test_track_attributes_set_after_adding_to_set(self):
        """
        Test that track attributes can be set after track is in ObjectTrackSet
        """
        track = self._create_track()
        obs = ObjectTrackSet([track])

        # Get track from set and set attributes
        retrieved_track = obs.get_track(0)
        retrieved_track.set_attribute("label", "object_1")
        retrieved_track.set_attribute("confidence", 0.95)

        # Verify attributes are accessible
        self.assertTrue(retrieved_track.has_attribute("label"))
        self.assertEqual(retrieved_track.get_attribute("label"), "object_1")
        self.assertAlmostEqual(retrieved_track.get_attribute("confidence"), 0.95)

    def test_track_attributes_via_tracks_list(self):
        """
        Test that track attributes work when accessing via tracks() list
        """
        track = self._create_track()
        track.set_attribute("track_type", "vehicle")

        obs = ObjectTrackSet([track])

        # Access track via tracks() list
        tracks_list = obs.tracks()
        self.assertEqual(len(tracks_list), 1)

        track_from_list = tracks_list[0]
        self.assertTrue(track_from_list.has_attribute("track_type"))
        self.assertEqual(track_from_list.get_attribute("track_type"), "vehicle")
