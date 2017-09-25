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

Interface to VITAL feature track class.

"""
import ctypes

from vital.types import (
    TrackState,
    Descriptor,
    Feature
)
from vital.util import VitalObject, free_void_ptr


class FeatureTrackState( TrackState ):
    """
    vital::track::feature_track_state interface class
    """
    def __init__(self, frame, feature=None, descriptor=None, from_cptr=None):
        """
        Initialize new track state

        :param feature: Optional Feature instance associated with this state.
        :type feature: vital.types.Feature

        :param descriptor: Optional Descriptor instance associated with this
            state.
        :type descriptor: vital.types.Descriptor
        """
        super(TrackState, self).__init__(from_cptr, frame, feature, descriptor)

    def _new(self, frame, feature, descriptor):
        """
        :param feature: Optional Feature instance associated with this state.
        :type feature: vital.types.Feature

        :param descriptor: Optional Descriptor instance associated with this
            state.
        :type descriptor: vital.types.Descriptor
        """
        return self._call_cfunc(
            "vital_feature_track_state_new",
            [ctypes.c_int64, Feature.c_ptr_type(), Descriptor.c_ptr_type()],
            [frame, feature, descriptor],
            self.C_TYPE_PTR
        )

    def _destroy(self):
        self._call_cfunc(
            "vital_track_state_destroy",
            [self.C_TYPE_PTR],
            [self],
        )

    @property
    def feature(self):
        f_ptr = self._call_cfunc(
            "vital_feature_track_state_feature",
            [self.C_TYPE_PTR],
            [self],
            Feature.c_ptr_type()
        )
        # f_ptr may be null
        if f_ptr:
            return Feature(from_cptr=f_ptr)
        else:
            return None

    @property
    def descriptor(self):
        d_ptr = self._call_cfunc(
            "vital_feature_track_state_descriptor",
            [self.C_TYPE_PTR],
            [self],
            Descriptor.c_ptr_type()
        )
        if d_ptr:
            return Descriptor(from_cptr=d_ptr)
        else:
            return None
