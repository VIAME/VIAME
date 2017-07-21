"""
ckwg +31
Copyright 2017 by Kitware, Inc.
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

Interface to VITAL object track class.

"""
import ctypes

from vital.types import (
    TrackStateData,
    DetectedObject
)
from vital.util import VitalObject, free_void_ptr


class ObjectTrackStateData (TrackStateData):
    """
    vital::track::feature_track_state_data interface class
    """
    def __init__(self, detection=None, from_cptr=None):
        """
        Initialize new track state

        :param detection: Optional DetectedObject instance associated with this state.
        :type detection: vital.types.DetectedObject
        """
        super(TrackState, self).__init__(from_cptr, detection)

    def _new(self, detection):
        """
        :param detection: Optional DetectedObject instance associated with this state.
        :type detection: vital.types.DetectedObject
        """
        return self._call_cfunc(
            "vital_object_track_state_data_new",
            [DetectedObject.c_ptr_type()],
            [detection],
            self.C_TYPE_PTR
        )

    def _destroy(self):
        self._call_cfunc(
            "vital_object_track_state_data_destroy",
            [self.C_TYPE_PTR],
            [self],
        )

    @property
    def detection(self):
        d_ptr = self._call_cfunc(
            "vital_object_track_state_data_detection",
            [self.C_TYPE_PTR],
            [self],
            DetectedObject.c_ptr_type()
        )
        # f_ptr may be null
        if d_ptr:
            return DetectedObject(from_cptr=d_ptr)
        else:
            return None

