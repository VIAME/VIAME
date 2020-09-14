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

import nose.tools
import numpy

from six.moves import range

from kwiver.vital.types import ObjectTrackSet, ObjectTrackState, BoundingBoxD as bbD, \
        DetectedObjectType as DOT, DetectedObject, Track


class TestObjectTrackSet (unittest.TestCase):
    def _create_track(self):
        """
        Helper function to create a track
        :return: Track with 10 object track state. Every track state has same
                    detected object however the fram number and time varies from
                    [0, 10)
        """
        bbox = bbD(10, 10, 20, 20)
        cm  = DOT("test", 0.4)
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
        nose.tools.assert_equal(obs.all_frame_ids(), set(range(10)))

    def test_first_frame(self):
        """
        Test first frame id in the track set is 0
        """
        obs = ObjectTrackSet([self._create_track()])
        nose.tools.assert_equal(obs.first_frame(), 0)

    def test_last_frame(self):
        """
        Test last frame id in the track set is 9
        """
        obs = ObjectTrackSet([self._create_track()])
        nose.tools.assert_equal(obs.last_frame(), 9)

    def test_tracks(self):
        """
        Test indexed retrival of tracks.
        """
        obj_track = self._create_track()
        obs = ObjectTrackSet([obj_track])
        nose.tools.assert_equal(obs.tracks()[0], obj_track)

    def test_get_track(self):
        """
        Test track retrival using track id
        """
        obj_track = self._create_track()
        obs = ObjectTrackSet([obj_track])
        nose.tools.assert_equal(obs.get_track(0), obj_track)
