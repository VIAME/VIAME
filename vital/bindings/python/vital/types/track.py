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

from vital.util import VitalObject, VitalErrorHandle


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
