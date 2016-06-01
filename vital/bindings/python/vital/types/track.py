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

Interface to VITAL track class.

"""
import ctypes

from vital.types import (
    Descriptor,
    Feature
)
from vital.util import VitalObject, free_void_ptr


class TrackState (VitalObject):
    """
    vital::track::track_state interface class
    """

    def __init__(self, frame, feature=None, descriptor=None, from_cptr=None):
        """
        Initialize new track state

        :param frame: Frame the track state intersects
        :type frame: int

        :param feature: Optional Feature instance associated with this state.
        :type feature: vital.types.Feature

        :param descriptor: Optional Descriptor instance associated with this
            state.
        :type descriptor: vital.types.Descriptor

        """
        super(TrackState, self).__init__(from_cptr, frame, feature, descriptor)

    def _new(self, frame, feature, descriptor):
        """
        :param frame: Frame the track state intersects
        :type frame: int

        :param feature: Optional Feature instance associated with this state.
        :type feature: vital.types.Feature

        :param descriptor: Optional Descriptor instance associated with this
            state.
        :type descriptor: vital.types.Descriptor
        """
        return self._call_cfunc(
            "vital_track_state_new",
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
    def frame_id(self):
        return self._call_cfunc(
            "vital_track_state_frame_id",
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_int64
        )

    @property
    def feature(self):
        f_ptr = self._call_cfunc(
            "vital_track_state_feature",
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
            "vital_track_state_descriptor",
            [self.C_TYPE_PTR],
            [self],
            Descriptor.c_ptr_type()
        )
        if d_ptr:
            return Descriptor(from_cptr=d_ptr)
        else:
            return None


class Track (VitalObject):
    """
    vital::track interface class
    """

    def __init__(self):
        """
        Initialize a new, empty track.
        """
        super(Track, self).__init__()

    def _new(self):
        return self._call_cfunc(
            'vital_track_new',
            restype=self.C_TYPE_PTR,
        )

    def _destroy(self):
        self._call_cfunc(
            'vital_track_destroy',
            [self.C_TYPE_PTR],
            [self]
        )

    def __len__(self):
        """
        :return: The number of states in this track
        :rtype: int
        """
        return self.size

    @property
    def id(self):
        """
        Get the ID of the track
        :return: Integer ID value of this track
        :rtype: int | long
        """
        return self._call_cfunc(
            "vital_track_id",
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_int64,
        )

    @id.setter
    def id(self, new_id):
        """
        Set ID of the track
        :param new_id: New integer ID
        :type new_id: int
        """
        self._call_cfunc(
            "vital_track_set_id",
            [self.C_TYPE_PTR, ctypes.c_int64],
            [self, new_id],
        )

    @property
    def first_frame(self):
        """
        Get the first frame ID of states in this track
        :return: Frame ID
        :rtype: int
        """
        return self._call_cfunc(
            "vital_track_first_frame",
            [self.C_TYPE_PTR], [self], ctypes.c_int64
        )

    @property
    def last_frame(self):
        """
        Get the last frame ID of states in this track
        :return: Frame ID
        :rtype: int
        """
        return self._call_cfunc(
            "vital_track_last_frame",
            [self.C_TYPE_PTR], [self], ctypes.c_int64
        )

    @property
    def size(self):
        """
        :return: The number of states in this track
        :rtype: int
        """
        return self._call_cfunc(
            "vital_track_size",
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_size_t,
        )

    @property
    def is_empty(self):
        """
        :return: If this track has no track states or not
        :rtype: bool
        """
        return self._call_cfunc(
            'vital_track_empty',
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_bool,
        )

    def all_frame_ids(self):
        """
        Get set of all frame IDs covered by states in this track.
        :return: Set of frame ID integers
        :rtype: set[int]
        """
        n = ctypes.c_size_t()
        s = self._call_cfunc(
            "vital_track_all_frame_ids",
            [self.C_TYPE_PTR, ctypes.POINTER(ctypes.c_size_t)],
            [self, ctypes.byref(n)],
            ctypes.POINTER(ctypes.c_int64)
        )
        r = set()
        for i in xrange(n.value):
            r.add(s[i])
        free_void_ptr(s)
        return r
