"""
ckwg +31
Copyright 2018 by Kitware, Inc.
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

Tests for ObjectTrackState

"""
import unittest

import nose.tools
import numpy

from kwiver.vital.types import ObjectTrackState, BoundingBox, DetectedObjectType, \
        DetectedObject


class TestObjectTrackState (unittest.TestCase):
    def _create_detected_object(self):
        """
        Helper function to generate a detected object for the track state
        :return: Detected object with bounding box coordinates of
                 (10, 10, 20, 20), confidence of 0.4 and "test" label
        """
        bbox = BoundingBox(10, 10, 20, 20)
        dot  = DetectedObjectType("test", 0.4)
        do = DetectedObject(bbox, 0.4, dot)
        return do

    def test_new_ts(self):
        """
        Test object track set creation with and without a detected object
        """
        do = self._create_detected_object()
        ObjectTrackState(0, 0, None)
        ObjectTrackState(0, 0, do)

    def test_frame_id(self):
        """
        Test frame id stored in a track state with >= 0 values
        """
        do = self._create_detected_object()
        ts = ObjectTrackState(0, 0, do)
        nose.tools.assert_equal(ts.frame_id, 0)
        ts = ObjectTrackState(14691234578, 0, do)
        nose.tools.assert_equal(ts.frame_id, 14691234578)

    def test_time_usec(self):
        """
        Test time in microsecond stored in a track state with >= 0 values
        """
        do = self._create_detected_object()
        ts = ObjectTrackState(0, 0, do)
        nose.tools.assert_equal(ts.time_usec, 0)
        ts = ObjectTrackState(0, 14691234578 , do)
        nose.tools.assert_equal(ts.time_usec, 14691234578)

    def test_detection(self):
        """
        Test detected object set in track state
        """
        do = self._create_detected_object()
        ts = ObjectTrackState(0, 0, do)
        nose.tools.assert_equal(ts.detection(), do)
