"""
ckwg +31
Copyright 2015-2016 by Kitware, Inc.
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

Tests for Python interface to vital::track_set

"""

from kwiver.vital.types import Track, TrackState
from kwiver.vital.types import TrackSet

import unittest


class TestVitalTrackSet(unittest.TestCase):
    def test_new(self):
        ts = TrackSet()

    def test_empty_len_size(self):
        ts = TrackSet()
        l = len(ts)
        s = ts.size()
        self.assertEqual(l, 0)
        self.assertEqual(l, s)

    def test_new_nonempty(self):
        n = 10
        tracks = [Track(i) for i in range(n)]
        ts = TrackSet(tracks)
        self.assertEqual(len(ts), n)
        self.assertEqual(ts.size(), n)

    def test_tracklist_accessor(self):
        n = 10
        tracks = [Track(i) for i in range(n)]
        ts = TrackSet(tracks)
        ts_tracks = ts.tracks()

        self.assertEqual(len(ts_tracks), n)
        for i, t in enumerate(ts_tracks):
            self.assertEqual(t.id, i)

    def test_all_frame_ids_single_track(self):
        # From a single track
        n = 10
        t = Track(1)
        for i in range(n):
            t.append(TrackState(i))
        ts = TrackSet([t])

        self.assertEqual(ts.all_frame_ids(), set(range(10)))

    def test_all_frame_ids_multitrack(self):
        # Across multiple tracks
        n = 10
        t1 = Track(1)
        for i in range(0, n):
            t1.append(TrackState(i))
        t2 = Track(2)
        for i in range(n, n + 5):
            t2.append(TrackState(i))
        ts = TrackSet([t1, t2])
        self.assertEqual(ts.all_frame_ids(), set(range(n + 5)))

    def test_first_frame(self):
        # no tracks
        ts = TrackSet()
        self.assertEqual(ts.first_frame(), 0)

        # one track
        t = Track(1)
        t.append(TrackState(1))
        t.append(TrackState(2))
        ts = TrackSet([t])
        self.assertEqual(ts.first_frame(), 1)

        # two tracks
        t2 = Track(2)
        t2.append(TrackState(3))
        ts = TrackSet([t, t2])
        self.assertEqual(ts.first_frame(), 1)

    def test_last_frame(self):
        # no tracks
        ts = TrackSet()
        self.assertEqual(ts.last_frame(), 0)

        # one track
        t = Track(1)
        t.append(TrackState(1))
        t.append(TrackState(2))
        ts = TrackSet([t])
        self.assertEqual(ts.last_frame(), 2)

        # two tracks
        t2 = Track(2)
        t2.append(TrackState(3))
        ts = TrackSet([t, t2])
        self.assertEqual(ts.last_frame(), 3)

    def test_get_track_empty(self):
        # Empty set
        ts = TrackSet()
        with self.assertRaises(IndexError):
            ts.get_track(0)

    def test_get_track_single(self):
        # get track when only track
        t = Track(0)
        ts = TrackSet([t])
        u = ts.get_track(0)

    def test_get_track_missing(self):
        # test failure to get track from set when set does have contents, but
        # TID is not present
        ts = TrackSet([Track(0), Track(1), Track(3)])
        with self.assertRaises(IndexError):
            ts.get_track(2)
