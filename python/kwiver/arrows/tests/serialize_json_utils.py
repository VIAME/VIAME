"""
ckwg +29
Copyright 2019-2020 by Kitware, Inc.
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

Utilities to support testing for json serialization of vital types
"""

from kwiver.vital.types import ActivityType
from kwiver.vital.types import BoundingBox
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObjectType
from kwiver.vital.types import ImageContainer, Image
from kwiver.vital.types import Timestamp
from kwiver.vital.types import TrackState
from kwiver.vital.types import ObjectTrackState
from kwiver.vital.types import Track
from kwiver.vital.types import TrackSet

def create_activity_type():
    return ActivityType("Test", 0.5)

def compare_activity_type(at1, at2):
    return at1.get_most_likely_class() == at2.get_most_likely_class() and \
            at1.get_most_likely_score() == at2.get_most_likely_score()

def create_bounding_box():
    return BoundingBox(1.0, 2.0, 3.0, 4.0)

def compare_bounding_box(bbox1, bbox2):
    return bbox1.min_x() == bbox2.min_x() and \
           bbox1.min_y() == bbox2.min_y() and \
           bbox1.max_x() == bbox2.max_x() and \
           bbox2.max_y() == bbox2.max_y()

def create_detected_object():
    return DetectedObject(create_bounding_box())

def compare_detected_object(do1, do2):
    return compare_bounding_box(do1.bounding_box(), do2.bounding_box())

def create_detected_object_set():
    dos = DetectedObjectSet()
    dos.add(DetectedObject(create_bounding_box()))
    return dos

def compare_detected_object_set(dos1, dos2):
    return len(dos1) == len(dos2) and \
           compare_detected_object(dos1[0], dos1[0])

def create_detected_object_type():
    return DetectedObjectType("Test", 0.25)

def compare_detected_object_type(dot1, dot2):
    return dot1.get_most_likely_class() == dot2.get_most_likely_class() and \
            dot1.get_most_likely_score() == dot2.get_most_likely_score()

def create_image():
    return ImageContainer(Image(720, 480))

def compare_image(img1, img2):
    return img1.width() == img2.width() and \
           img1.height() == img2.height()

def create_timestamp():
    return Timestamp(10, 20)

def compare_timestamp(ts1, ts2):
    return ts1.get_frame() == ts2.get_frame() and \
            ts1.get_time_usec() == ts2.get_time_usec()

def create_track_state():
    return TrackState(10)

def compare_track_state(ts1, ts2):
    return ts1.frame_id == ts2.frame_id

def create_object_track_state():
    return ObjectTrackState(10, 15, create_detected_object())

def compare_object_track_state(ots1, ots2):
    return ts1.frame_id == ts2.frame_id and \
           ts1.time_usec == ts2.time_usec and \
           compare_detected_object(ots1.detection(), ots2.detection())

def create_track():
    track = Track()
    track.append(create_track_state())
    return track

def compare_track(t1, t2):
    return len(t1) == len(t2) and \
            compare_track_state(t1.find_state(10), t2.find_state(10))

def create_track_set():
    return TrackSet([create_track()])


def compare_track_set(ts1, ts2):
    return compare_track(ts1.tracks()[0], ts2.tracks()[0])

def create_object_track_set():
    object_track_state = create_object_track_state()
    track = Track()
    track.append(create_track_state())
    return ObjectTrackSet([track])

def compare_object_track(ot1, ot2):
    return len(ot1) == len(ot2) and \
            compare_object_track_state(ot1.find_state(10), ot2.find_state(10))

def compare_object_track_set(ots1, ots2):
    return compare_object_track(ots1.tracks()[0], ots2.track()[0])
