"""
ckwg +31
Copyright 2015-2017 by Kitware, Inc.
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

from vital.util import VitalObject, free_void_ptr


class TrackStateData (VitalObject):
    """
    vital::track::track_state_data interface class
    
    Note this is an empty base class, all the work done with
    interfacing is done in derived versions of this class.
    """

    def __init__(self, from_cptr=None):
        """
        Initialize new track state data
        """
        super(TrackStateData, self).__init__(from_cptr)


class TrackState (VitalObject):
    """
    vital::track::track_state interface class
    """

    def __init__(self, frame=0, data=None, from_cptr=None):
        """
        Initialize new track state

        :param frame: Frame the track state intersects
        :type frame: int

        :param data: Optional data instance associated with this state.
        :type data: vital.types.TrackStateData
        """
        super(TrackState, self).__init__(from_cptr, frame, data)

    def _new(self, frame, data):
        """
        :param frame: Frame the track state intersects
        :type frame: int

        :param data: Optional Data instance associated with this state.
        :type TrackStateData: vital.types.TrackStateData
        """
        return self._call_cfunc(
            "vital_track_state_new",
            [ctypes.c_int64, TrackStateData.c_ptr_type()],
            [frame, data],
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


class Track (VitalObject):
    """
    vital::track interface class

    Track states can be yielded by iterating over this object, or by fetching
    all contained frame IDs and iteratively requesting each state individually.

    """

    def __init__(self, id=0, from_cptr=None):
        """
        Initialize a new, empty track.

        :param id: ID number to assign to this track

        """
        super(Track, self).__init__(from_cptr)
        # Set given ID value after construction if not from an existing pointer
        if id != 0 and from_cptr is None:
            self.id = id

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

    def __getitem__(self, fid):
        """
        Get the track state matching the given frame ID

        :param fid: the frame ID to look for among states
        :type fid: int

        :return: TrackState instance from this track that intersects the given
            frame id.
        :rtype: TrackState

        :raises IndexError: The given frame ID is not covered by states in this
            track.

        """
        return self.find_state(fid)

    def __iter__(self):
        """
        Iterate through TrackStates in this Track by ordered frame ID.
        :rtype: __generator[TrackState]
        """
        for fid in sorted(self.all_frame_ids()):
            yield self.find_state(fid)

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

    def append(self, ts):
        """
        Append a track state to this track

        The new track state must have a frame_id greater than the last frame in
        the history. If such an append is attempted, nothing is added to this
        track.

        :param ts: TrackState instance to add to this track.
        :type ts: TrackState

        :return: True if the state was successfully added, False if it wasn't.
            If False is returned, this track is no modified.
        :rtype: bool

        """
        return self._call_cfunc(
            'vital_track_append_state',
            [Track.c_ptr_type(), TrackState.c_ptr_type()],
            [self, ts],
            ctypes.c_bool
        )

    def find_state(self, frame_id):
        """
        Find the track state matching the given frame ID

        :param frame_id: the frame ID to look for among states
        :type frame_id: int

        :return: TrackState instance from this track that intersects the given
            frame id.
        :rtype: TrackState

        :raises IndexError: The given frame ID is not covered by states in this
            track.

        """
        ts_cptr = self._call_cfunc(
            'vital_track_find_state',
            [self.C_TYPE_PTR, ctypes.c_int64],
            [self, frame_id],
            TrackState.c_ptr_type()
        )
        if ts_cptr:
            return TrackState(from_cptr=ts_cptr)
        else:
            raise IndexError(frame_id)
