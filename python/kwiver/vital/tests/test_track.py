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

import numpy
import unittest

from kwiver.vital.types import Track, TrackState, Feature, Descriptor


class TestVitalTrack(unittest.TestCase):
    def test_new(self):
        t = Track()

    def test_initial_id(self):
        t = Track()
        self.assertEqual(t.id, 0)

        t = Track(0)
        self.assertEqual(t.id, 0)

        t = Track(-1)
        self.assertEqual(t.id, -1)

        t = Track(15)
        self.assertEqual(t.id, 15)

    def test_initial_firstlast_frame(self):
        t = Track()
        self.assertEqual(t.first_frame, 0)
        self.assertEqual(t.last_frame, 0)

    def test_initial_all_frame_ids(self):
        t = Track()
        s = t.all_frame_ids()
        self.assertEqual(len(s), 0)

    def test_initial_size(self):
        t = Track()
        self.assertEqual(t.size, 0)
        self.assertEqual(len(t), 0)

    def test_initial_is_empty(self):
        t = Track()
        self.assertTrue(t.is_empty)

    def test_set_id(self):
        t = Track()
        self.assertEqual(t.id, 0)

        t.id = 2
        self.assertEqual(t.id, 2)

        t.id = 1345634
        self.assertEqual(t.id, 1345634)

    def test_ts_append(self):
        t = Track()
        self.assertEqual(t.size, 0)
        self.assertEqual(len(t), 0)

        ts = TrackState(10)
        self.assertTrue(t.append(ts))
        self.assertEqual(t.size, 1)
        self.assertEqual(len(t), 1)

        ts = TrackState(11)
        self.assertTrue(t.append(ts))
        self.assertEqual(t.size, 2)
        self.assertEqual(len(t), 2)

        # Other properties that should not be different than default
        self.assertEqual(t.first_frame, 10)
        self.assertEqual(t.last_frame, 11)
        self.assertFalse(t.is_empty)

    def test_ts_append_outoforder(self):
        t = Track()
        self.assertTrue(t.append(TrackState(10)))
        self.assertFalse(t.append(TrackState(10)))
        self.assertFalse(t.append(TrackState(9)))
        self.assertFalse(t.append(TrackState(0)))
        self.assertFalse(t.append(TrackState(-1)))

        self.assertTrue(t.append(TrackState(11)))
        self.assertFalse(t.append(TrackState(11)))

        # After all that there should only be two states in there for frames 10
        # and 11.
        self.assertEqual(t.size, 2)
        self.assertEqual(len(t), 2)
        self.assertEqual(t.all_frame_ids(), {10, 11})

    def test_track_find(self):
        t = Track()
        t.append(TrackState(0))
        t.append(TrackState(1))
        t.append(TrackState(5))
        t.append(TrackState(9))

        ts = t.find_state(0)
        self.assertIsNotNone(ts)
        self.assertEqual(ts.frame_id, 0)

        ts = t.find_state(1)
        self.assertIsNotNone(ts)
        self.assertEqual(ts.frame_id, 1)

        ts = t.find_state(5)
        self.assertIsNotNone(ts)
        self.assertEqual(ts.frame_id, 5)

        ts = t.find_state(9)
        self.assertIsNotNone(ts)
        self.assertEqual(ts.frame_id, 9)

        with self.assertRaises(IndexError):
            t.find_state(10)
        t.append(TrackState(10))
        self.assertIsNotNone(t.find_state(10))
        self.assertEqual(t.find_state(10).frame_id, 10)

    def test_track_getitem(self):
        # this is the same test as test_track_find, but using the get-item
        # accessor syntax
        t = Track()
        t.append(TrackState(0))
        t.append(TrackState(1))
        t.append(TrackState(5))
        t.append(TrackState(9))

        ts = t[0]
        self.assertIsNotNone(ts)
        self.assertEqual(ts.frame_id, 0)

        ts = t[1]
        self.assertIsNotNone(ts)
        self.assertEqual(ts.frame_id, 1)

        ts = t[5]
        self.assertIsNotNone(ts)
        self.assertEqual(ts.frame_id, 5)

        ts = t[9]
        self.assertIsNotNone(ts)
        self.assertEqual(ts.frame_id, 9)

        with self.assertRaises(IndexError):
            t.find_state(10)
        t.append(TrackState(10))
        self.assertIsNotNone(t[10])
        self.assertEqual(t[10].frame_id, 10)

    def test_iteration(self):
        t = Track()
        t.append(TrackState(0))
        t.append(TrackState(1))
        t.append(TrackState(5))
        t.append(TrackState(9))

        self.assertEqual([ts.frame_id for ts in t], [0, 1, 5, 9])

    def test_has_attribute_empty(self):
        t = Track()
        self.assertFalse(t.has_attribute("nonexistent"))

    def test_set_get_attribute_string(self):
        t = Track()
        t.set_attribute("name", "test_track")
        self.assertTrue(t.has_attribute("name"))
        self.assertEqual(t.get_attribute("name"), "test_track")

    def test_set_get_attribute_int(self):
        t = Track()
        t.set_attribute("count", 42)
        self.assertTrue(t.has_attribute("count"))
        self.assertEqual(t.get_attribute("count"), 42)

    def test_set_get_attribute_float(self):
        t = Track()
        t.set_attribute("score", 0.95)
        self.assertTrue(t.has_attribute("score"))
        self.assertAlmostEqual(t.get_attribute("score"), 0.95)

    def test_set_get_attribute_bool(self):
        t = Track()
        t.set_attribute("verified", True)
        self.assertTrue(t.has_attribute("verified"))
        self.assertEqual(t.get_attribute("verified"), True)

    def test_attribute_keys(self):
        t = Track()
        self.assertEqual(t.attribute_keys(), [])

        t.set_attribute("name", "test")
        t.set_attribute("count", 10)
        t.set_attribute("score", 0.5)

        keys = t.attribute_keys()
        self.assertEqual(len(keys), 3)
        self.assertIn("name", keys)
        self.assertIn("count", keys)
        self.assertIn("score", keys)

    def test_get_attribute_nonexistent(self):
        t = Track()
        with self.assertRaises(RuntimeError):
            t.get_attribute("nonexistent")

    def test_attribute_overwrite(self):
        t = Track()
        t.set_attribute("value", 10)
        self.assertEqual(t.get_attribute("value"), 10)
        t.set_attribute("value", 20)
        self.assertEqual(t.get_attribute("value"), 20)
