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
import ctypes
from six.moves import range

from vital.types import Track, TrackState
from vital.types import TrackSet

import nose.tools as nt


class TestVitalTrackSet (object):

    def test_new(self):
        ts = TrackSet()
        nt.assert_true(ts, "Invalid track set instance constructed")

    def test_empty_len_size(self):
        ts = TrackSet()
        nt.assert_true(ts, "Invalid track set instance constructed")
        l = len(ts)
        s = ts.size()
        nt.assert_equal(l, 0)
        nt.assert_equal(l, s)

    def test_new_nonempty(self):
        n = 10
        tracks = [Track(i) for i in range(n)]
        ts = TrackSet(tracks)
        nt.assert_true(ts, "Invalid track set instance constructed")
        nt.assert_equal(len(ts), n)
        nt.assert_equal(ts.size(), n)

    def test_tracklist_accessor(self):
        n = 10
        tracks = [Track(i) for i in range(n)]
        ts = TrackSet(tracks)
        ts_tracks = ts.tracks()

        nt.assert_equal(len(ts_tracks), n)
        for i, t in enumerate(ts_tracks):
            nt.assert_equal(t.id, i)

        # constructing a new trackset from the returned list of tracks should
        # yield a trackset whose accessor should return the same list of tracks
        # (same C/C++ instances).
        ts2 = TrackSet(ts_tracks)
        ts2_tracks = ts2.tracks()
        for i in range(n):
            ctypes.addressof(ts2_tracks[0].c_pointer.contents) \
                == ctypes.addressof(ts2_tracks[0].c_pointer.contents)

    def test_all_frame_ids_single_track(self):
        # From a single track
        n = 10
        t = Track(1)
        for i in range(n):
            t.append(TrackState(i))
        ts = TrackSet([t])

        nt.assert_equal(
            ts.all_frame_ids(),
            set(range(10))
        )

    def test_all_frame_ids_multitrack(self):
        # Across multiple tracks
        n = 10
        t1 = Track(1)
        for i in range(0, n):
            t1.append(TrackState(i))
        t2 = Track(2)
        for i in range(n, n+5):
            t2.append(TrackState(i))
        ts = TrackSet([t1, t2])
        nt.assert_equal(
            ts.all_frame_ids(),
            set(range(n+5))
        )

    def test_first_frame(self):
        # no tracks
        ts = TrackSet()
        nt.assert_equal(ts.first_frame(), 0)

        # one track
        t = Track(1)
        t.append(TrackState(1))
        t.append(TrackState(2))
        ts = TrackSet([t])
        nt.assert_equal(ts.first_frame(), 1)

        # two tracks
        t2 = Track(2)
        t2.append(TrackState(3))
        ts = TrackSet([t, t2])
        nt.assert_equal(ts.first_frame(), 1)

    def test_last_frame(self):
        # no tracks
        ts = TrackSet()
        nt.assert_equal(ts.last_frame(), 0)

        # one track
        t = Track(1)
        t.append(TrackState(1))
        t.append(TrackState(2))
        ts = TrackSet([t])
        nt.assert_equal(ts.last_frame(), 2)

        # two tracks
        t2 = Track(2)
        t2.append(TrackState(3))
        ts = TrackSet([t, t2])
        nt.assert_equal(ts.last_frame(), 3)

    def test_get_track_empty(self):
        # Empty set
        ts = TrackSet()
        nt.assert_raises(
            IndexError,
            ts.get_track, 0
        )

    def test_get_track_single(self):
        # get track when only track
        t = Track(0)
        ts = TrackSet([t])
        u = ts.get_track(0)
        # Check that they're the same underlying instance
        nt.assert_equal(
            ctypes.addressof(t.c_pointer.contents),
            ctypes.addressof(u.c_pointer.contents)
        )

    def test_get_track_missing(self):
        # test failure to get track from set when set does have contents, but
        # TID is not present
        ts = TrackSet([Track(0), Track(1), Track(3)])
        nt.assert_raises(
            IndexError,
            ts.get_track, 2
        )

    def test_get_track_multiple(self):
        n = 10
        tracks = [Track(i) for i in range(n)]
        ts = TrackSet(tracks)

        # Check that they're the same underlying instance
        for i in range(n):
            nt.assert_equal(
                ctypes.addressof(ts.get_track(i).c_pointer.contents),
                ctypes.addressof(tracks[i].c_pointer.contents)
            )
