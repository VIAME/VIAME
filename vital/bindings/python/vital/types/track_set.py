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
__author__ = 'paul.tunison@kitware.com'

import ctypes

from vital.types import Track
from vital.util import VitalObject, VitalErrorHandle


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
        ts_new_from_file = cls.VITAL_LIB['vital_trackset_new_from_file']
        ts_new_from_file.argtypes = [ctypes.c_char_p,
                                     VitalErrorHandle.C_TYPE_PTR]
        ts_new_from_file.restype = cls.C_TYPE_PTR

        with VitalErrorHandle() as eh:
            return TrackSet(from_cptr=ts_new_from_file(filepath, eh))

    def __init__(self, track_list=None, from_cptr=None):
        """
        Create a new track set from a list of tracks.

        None or an empty list may be provided to initialize an empty track set.

        :param track_list: List of tracks to initialize the set with
        :type track_list: list of Track or None

        """
        super(TrackSet, self).__init__(from_cptr, track_list)

    def _new(self, track_list):
        ts_new = self.VITAL_LIB['vital_trackset_new']
        ts_new.argtypes = [ctypes.c_size_t,
                           ctypes.POINTER(Track.C_TYPE_PTR)]
        ts_new.restype = self.C_TYPE_PTR

        if track_list is None:
            track_list = []
        # noinspection PyCallingNonCallable
        c_track_array = (Track.C_TYPE_PTR * len(track_list))(
            *(t.c_pointer for t in track_list)
        )
        return ts_new(len(track_list), c_track_array)

    def _destroy(self):
        ts_del = self.VITAL_LIB['vital_trackset_destroy']
        ts_del.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            ts_del(self, eh)

    def __len__(self):
        ts_size = self.VITAL_LIB['vital_trackset_size']
        ts_size.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        ts_size.restype = ctypes.c_size_t
        with VitalErrorHandle() as eh:
            return ts_size(self, eh)

    def size(self):
        return len(self)

    def write_tracks_file(self, filepath):
        """
        Write this track set to the given filepath

        :param filepath: The path to the file to write to.
        :type filepath: str

        """
        ts_write = self.VITAL_LIB['vital_trackset_write_track_file']
        ts_write.argtypes = [self.C_TYPE_PTR, ctypes.c_char_p,
                             VitalErrorHandle.C_TYPE_PTR]

        with VitalErrorHandle() as eh:
            ts_write(self, filepath, eh)
