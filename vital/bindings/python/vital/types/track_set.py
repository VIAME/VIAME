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

Interface to VITAL track_set class.

"""
# -*- coding: utf-8 -*-

import ctypes

from vital.types import Track
from vital.util import VitalObject, free_void_ptr


class TrackSet (VitalObject):
    """
    vital::track_set interface class
    """

    @classmethod
    def from_file(cls, filepath):
        """
        Create a new track set as read from the given filepath.
        :param filepath: Path to a file to a track set from
        :type filepath: str

        :return: New track set as read from the given file.
        :rtype: TrackSet

        """
        cptr = cls._call_cfunc(
            'vital_trackset_new_from_file',
            [ctypes.c_char_p],
            [filepath],
            cls.c_ptr_type()
        )
        return TrackSet(from_cptr=cptr)

    def __init__(self, track_list=None, from_cptr=None):
        """
        Create a new track set from a list of tracks.

        None or an empty list may be provided to initialize an empty track set.

        :param track_list: List of tracks to initialize the set with
        :type track_list: collections.Iterable[Track] | None

        """
        super(TrackSet, self).__init__(from_cptr, track_list)

    def _new(self, track_list):
        """
        :param track_list: List of tracks to initialize the set with
        :type track_list: collections.Iterable[Track] | None
        """
        if track_list is None:
            track_list = []
        # noinspection PyCallingNonCallable
        c_track_array = (Track.c_ptr_type() * len(track_list))(
            *(t.c_pointer for t in track_list)
        )

        return self._call_cfunc(
            'vital_trackset_new',
            [ctypes.c_size_t, ctypes.POINTER(Track.c_ptr_type())],
            [len(track_list), c_track_array],
            self.C_TYPE_PTR
        )

    def _destroy(self):
        self._call_cfunc(
            'vital_trackset_destroy',
            [self.C_TYPE_PTR],
            [self]
        )

    def __len__(self):
        return self.size()

    def __iter__(self):
        for tid in self.all_track_ids():
            yield self.get_track(tid)

    def size(self):
        return self._call_cfunc(
            'vital_trackset_size',
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_size_t
        )

    def tracks(self):
        """
        Get the list of all tracks contained in this set (new instances).

        :return: list of new Track instances of tracks contained in this set
        :rtype: list[Track]

        """
        c_ptr_arr = self._call_cfunc(
            'vital_trackset_tracks',
            [self.C_TYPE_PTR],
            [self],
            ctypes.POINTER(Track.c_ptr_type())
        )

        tracks = []
        for cptr in c_ptr_arr[:self.size()]:
            cptr = Track.c_ptr_type()(cptr.contents)
            tracks.append(Track(from_cptr=cptr))
        free_void_ptr(c_ptr_arr)

        return tracks

    def all_frame_ids(self):
        """
        Get the set of all frame IDs covered by tracks in this set

        :return: Set of frame IDs covered
        :rtype: set[int]

        """
        arr_len = ctypes.c_size_t()
        fid_arr = self._call_cfunc(
            'vital_trackset_all_frame_ids',
            [self.C_TYPE_PTR, ctypes.POINTER(ctypes.c_size_t)],
            [self, ctypes.byref(arr_len)],
            ctypes.POINTER(ctypes.c_int64)
        )
        fid_set = set()
        for i in range(arr_len.value):
            fid_set.add(fid_arr[i])
        free_void_ptr(fid_arr)
        return fid_set

    def all_track_ids(self):
        """ Get the set of all track IDs contained in this set
        :return: Set of track ID integers
        :rtype: set[int]
        """
        arr_len = ctypes.c_size_t()
        tid_arr = self._call_cfunc(
            'vital_trackset_all_track_ids',
            [self.C_TYPE_PTR, ctypes.POINTER(ctypes.c_size_t)],
            [self, ctypes.byref(arr_len)],
            ctypes.POINTER(ctypes.c_int64)
        )
        tid_set = set()
        for i in range(arr_len.value):
            tid_set.add(tid_arr[i])
        free_void_ptr(tid_arr)
        return tid_set

    def first_frame(self):
        """ Get the first (smallest) frame number containing tracks

        If there are no tracks in this set, or the contained tracks have no
        states, this returns 0.

        :return frame ID of the smallest overlapping frame
        :rtype: int
        """
        return self._call_cfunc(
            'vital_trackset_first_frame',
            [self.C_TYPE_PTR], [self],
            ctypes.c_int64
        )

    def last_frame(self):
        """ Get the last (largest) frame number containing tracks

        If there are no tracks in this set, or the contained tracks have no
        states, this returns 0.

        :return: frame ID of the largest overlapping frame
        :rtype: int
        """
        return self._call_cfunc(
            'vital_trackset_last_frame',
            [self.C_TYPE_PTR], [self],
            ctypes.c_int64
        )

    def get_track(self, tid):
        """ Get the track in this set with the specified ID

        :raises IndexError: If no tracks in this set match the given ID.

        :param tid: The ID of the track to get
        :type tid: int

        :return: New Track instance referring to the track with the given ID in
            this set, or None if no tracks in this set have the given ID.
        :rtype: Track

        """
        track_cptr = self._call_cfunc(
            'vital_trackset_get_track',
            [self.C_TYPE_PTR, ctypes.c_int64], [self, tid],
            Track.c_ptr_type()
        )
        if not track_cptr:
            raise IndexError(tid)
        return Track(from_cptr=track_cptr)

    def write_tracks_file(self, filepath):
        """
        Write this track set to the given filepath

        :param filepath: The path to the file to write to.
        :type filepath: str

        """
        self._call_cfunc(
            'vital_trackset_write_track_file',
            [self.C_TYPE_PTR, ctypes.c_char_p],
            [self, filepath]
        )
