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

Tests for Track interface class

"""

import nose.tools
import numpy

from kwiver.vital.types import Track, TrackState, Feature, Descriptor


class TestVitalTrack (object):

    def test_new(self):
        t = Track()

    def test_initial_id(self):
        t = Track()
        nose.tools.assert_equal(t.id, 0)

        t = Track(0)
        nose.tools.assert_equal(t.id, 0)

        t = Track(-1)
        nose.tools.assert_equal(t.id, -1)

        t = Track(15)
        nose.tools.assert_equal(t.id, 15)

    def test_initial_firstlast_frame(self):
        t = Track()
        nose.tools.assert_equal(t.first_frame, 0)
        nose.tools.assert_equal(t.last_frame, 0)

    def test_initial_all_frame_ids(self):
        t = Track()
        s = t.all_frame_ids()
        nose.tools.assert_equal(len(s), 0)

    def test_initial_size(self):
        t = Track()
        nose.tools.assert_equal(t.size, 0)
        nose.tools.assert_equal(len(t), 0)

    def test_initial_is_empty(self):
        t = Track()
        nose.tools.assert_true(t.is_empty)

    def test_set_id(self):
        t = Track()
        nose.tools.assert_equal(t.id, 0)

        t.id = 2
        nose.tools.assert_equal(t.id, 2)

        t.id = 1345634
        nose.tools.assert_equal(t.id, 1345634)


    def test_ts_append(self):
        t = Track()
        nose.tools.assert_equal(t.size, 0)
        nose.tools.assert_equal(len(t), 0)

        ts = TrackState(10)
        nose.tools.assert_true(t.append(ts))
        nose.tools.assert_equal(t.size, 1)
        nose.tools.assert_equal(len(t), 1)

        ts = TrackState(11)
        nose.tools.assert_true(t.append(ts))
        nose.tools.assert_equal(t.size, 2)
        nose.tools.assert_equal(len(t), 2)

        # Other properties that should not be different than default
        nose.tools.assert_equal(t.first_frame, 10)
        nose.tools.assert_equal(t.last_frame, 11)
        nose.tools.assert_false(t.is_empty)

    def test_ts_append_outoforder(self):
        t = Track()
        nose.tools.assert_true(t.append(TrackState(10)))
        nose.tools.assert_false(t.append(TrackState(10)))
        nose.tools.assert_false(t.append(TrackState(9)))
        nose.tools.assert_false(t.append(TrackState(0)))
        nose.tools.assert_false(t.append(TrackState(-1)))

        nose.tools.assert_true(t.append(TrackState(11)))
        nose.tools.assert_false(t.append(TrackState(11)))

        # After all that there should only be two states in there for frames 10
        # and 11.
        nose.tools.assert_equal(t.size, 2)
        nose.tools.assert_equal(len(t), 2)
        nose.tools.assert_equal(t.all_frame_ids(), {10, 11})

    def test_track_find(self):
        t = Track()
        t.append(TrackState(0))
        t.append(TrackState(1))
        t.append(TrackState(5))
        t.append(TrackState(9))

        ts = t.find_state(0)
        nose.tools.assert_is_not_none(ts)
        nose.tools.assert_equal(ts.frame_id, 0)

        ts = t.find_state(1)
        nose.tools.assert_is_not_none(ts)
        nose.tools.assert_equal(ts.frame_id, 1)

        ts = t.find_state(5)
        nose.tools.assert_is_not_none(ts)
        nose.tools.assert_equal(ts.frame_id, 5)

        ts = t.find_state(9)
        nose.tools.assert_is_not_none(ts)
        nose.tools.assert_equal(ts.frame_id, 9)

        nose.tools.assert_raises(
            IndexError,
            t.find_state, 10
        )
        t.append(TrackState(10))
        nose.tools.assert_is_not_none(t.find_state(10))
        nose.tools.assert_equal(t.find_state(10).frame_id, 10)

    def test_track_getitem(self):
        # this is the same test as test_track_find, but using the get-item
        # accessor syntax
        t = Track()
        t.append(TrackState(0))
        t.append(TrackState(1))
        t.append(TrackState(5))
        t.append(TrackState(9))

        ts = t[0]
        nose.tools.assert_is_not_none(ts)
        nose.tools.assert_equal(ts.frame_id, 0)

        ts = t[1]
        nose.tools.assert_is_not_none(ts)
        nose.tools.assert_equal(ts.frame_id, 1)

        ts = t[5]
        nose.tools.assert_is_not_none(ts)
        nose.tools.assert_equal(ts.frame_id, 5)

        ts = t[9]
        nose.tools.assert_is_not_none(ts)
        nose.tools.assert_equal(ts.frame_id, 9)

        nose.tools.assert_raises(
            IndexError,
            t.find_state, 10
        )
        t.append(TrackState(10))
        nose.tools.assert_is_not_none(t[10])
        nose.tools.assert_equal(t[10].frame_id, 10)

    def test_iteration(self):
        t = Track()
        t.append(TrackState(0))
        t.append(TrackState(1))
        t.append(TrackState(5))
        t.append(TrackState(9))

        nose.tools.assert_equal(
            [ts.frame_id for ts in t],
            [0, 1, 5, 9]
        )
