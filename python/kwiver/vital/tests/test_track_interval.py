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

Tests for track_interval interface

"""

from kwiver.vital.types import TrackInterval, Timestamp
import unittest


class TestVitalTrackInterval(unittest.TestCase):
    def test_create(self):
        TrackInterval()
        TrackInterval(20, Timestamp(), Timestamp())
        TrackInterval(21, Timestamp(1234, 1), Timestamp(5678, 2))

    def test_get_set_track_id(self):
        ti = TrackInterval()
        self.assertEqual(ti.track, 0)

        ti.track = 5
        self.assertEqual(ti.track, 5)

        ti.track = -12
        self.assertEqual(ti.track, -12)

        ti.track = 0
        self.assertEqual(ti.track, 0)

        # Check initial id
        ti = TrackInterval(20, Timestamp(), Timestamp())
        self.assertEqual(ti.track, 20)

        ti.track = 5
        self.assertEqual(ti.track, 5)

        ti.track = -12
        self.assertEqual(ti.track, -12)

        ti.track = 0
        self.assertEqual(ti.track, 0)

    def test_get_set_timestamps(self):
        ti = TrackInterval()
        self.assertFalse(ti.start.is_valid())
        self.assertFalse(ti.stop.is_valid())

        ts1, ts2 = Timestamp(1234, 1), Timestamp(5678, 2)
        ti.start = ts1
        ti.stop = ts2
        self.assertTrue(ti.start == ts1)
        self.assertTrue(ti.stop == ts2)

        # Confirm its a copy, not a reference
        ts1.set_frame(3)
        self.assertTrue(ti.start != ts1)

        ti.start.set_frame(3)
        self.assertTrue(ti.start == ts1)

        # Getting and setting with other constructor
        ti = TrackInterval(21, Timestamp(1234, 1), Timestamp(5678, 2))
        self.assertTrue(ti.start.is_valid())
        self.assertTrue(ti.stop.is_valid())
        self.assertTrue(ti.start == Timestamp(1234, 1))
        self.assertTrue(ti.stop == Timestamp(5678, 2))

        ti.stop = Timestamp()
        self.assertFalse(ti.stop.is_valid())
        ti.stop.set_time_seconds(4321)
        self.assertEqual(ti.stop.get_time_seconds(), 4321)

        ts1 = Timestamp(8765, 4)
        ti.start = ts1
        self.assertTrue(ti.start == ts1)

    def test_set_incorrect_type(self):
        ti = TrackInterval()

        with self.assertRaises(TypeError):
            ti.track = "5"

        with self.assertRaises(TypeError):
            ti.start = "5"

        with self.assertRaises(TypeError):
            ti.stop = "5"

        ti = TrackInterval(21, Timestamp(1234, 1), Timestamp(5678, 2))

        with self.assertRaises(TypeError):
            ti.track = "5"

        with self.assertRaises(TypeError):
            ti.start = "5"

        with self.assertRaises(TypeError):
            ti.stop = "5"

    def test_set_no_attribute(self):
        ti = TrackInterval()
        with self.assertRaises(AttributeError):
            ti.nonexistant_attribute = 5

        ti = TrackInterval(21, Timestamp(1234, 1), Timestamp(5678, 2))
        with self.assertRaises(AttributeError):
            ti.nonexistant_attribute = 5
