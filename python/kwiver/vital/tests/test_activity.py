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

Tests for Activity python class.

"""

import unittest
import nose.tools as nt
import numpy as np
from kwiver.vital.types import ( Activity,
            ActivityType,
            DetectedObjectType,
            Timestamp,
            BoundingBoxD as BoundingBox,
            DetectedObject,
            ObjectTrackState,
            ObjectTrackSet,
            Track,
            )

class TestActivity(unittest.TestCase):
    @classmethod
    def setUp(self):
        bbox = BoundingBox(10, 10, 20, 20)
        dot  = DetectedObjectType("test", 0.4)
        do = DetectedObject(bbox, 0.4, dot)
        track = Track()
        for i in range(10):
            track.append(ObjectTrackState(i, i, do))
        self.track_ = track
        self.time_1 = Timestamp()
        self.time_1.set_time_seconds(1234)
        self.time_2 = Timestamp()
        self.time_2.set_time_seconds(4321)
        self.obj_ts = ObjectTrackSet([self.track_])
        self.act_type = ActivityType("self_act", 0.87)
        self.act = Activity(1, "self_act", 0.87, self.act_type, self.time_1, self.time_2, self.obj_ts)
    def test_constructors(self):
        Activity()
        Activity(1, "first_act", 0.87, ActivityType(), self.time_1, self.time_2)
    def test_id(self):
        a = self.act
        self.assertEqual(a.id, 1)
        a.id = 10
        self.assertEqual(a.id, 10)
        self.assertEqual(a.label, "self_act")
        a.label = "second_act"
        self.assertEqual(a.label, "second_act")
        self.assertEqual(a.activity_type.score("self_act"), 0.87)
        a.activity_type = ActivityType()
        self.assertEqual(a.confidence, 0.87)
        a.confidence = 1
        self.assertEqual(a.confidence, 1)
        self.assertEqual(a.start_time.get_time_seconds(), 1234)
        tmp_time = Timestamp().set_time_seconds(1237)
        a.start_time = tmp_time
        self.assertEqual(a.start_time.get_time_seconds(), 1237)
        self.assertEqual(a.end_time.get_time_seconds(), 4321)
        tmp_time = Timestamp()
        tmp_time.set_time_seconds(4322)
        a.end_time = tmp_time
        self.assertEqual(a.end_time.get_time_seconds(), 4322)
        self.assertEqual(a.participants.all_frame_ids(), set(range(10)))
        bbox = BoundingBox(10, 10, 20, 20)
        dot  = DetectedObjectType("test", 0.4)
        do = DetectedObject(bbox, 0.4, dot)
        track = Track()
        for i in range(5):
            track.append(ObjectTrackState(i, i, do))
        new_t = track
        new_ots = ObjectTrackSet([new_t])
        a.participants = new_ots
        self.assertEqual(a.participants.all_frame_ids(), set(range(5)))
        self.assertEqual(a.duration[0].get_time_seconds(), a.start_time.get_time_seconds())
        self.assertEqual(a.duration[1].get_time_seconds(), a.end_time.get_time_seconds())
