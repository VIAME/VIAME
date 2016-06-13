"""
ckwg +31
Copyright 2016 by Kitware, Inc.
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

vital::landmark_map interface

"""
import ctypes

from vital.types import Landmark
from vital.util import VitalObject, free_void_ptr


class LandmarkMap (VitalObject):

    @classmethod
    def from_dict(cls, id_lm_d):
        """
        Create a new instance of LandmarkMap using the given dictionary mapping
        integer IDs to Landmark instances.

        :param id_lm_d: dictionary mapping integer IDs to Landmark instances
        :type id_lm_d: dict[int|long, vital.types.Landmark]

        :return: New landmark map instance containing a copy of the input map.
        :rtype: LandmarkMap

        """
        s = len(id_lm_d)
        t_lm_ids = (ctypes.c_int64 * s)
        t_lm_landmarks = (Landmark.c_ptr_type() * s)

        lm_ids = t_lm_ids()
        lm_landmarks = t_lm_landmarks()
        i = 0
        for k, l in id_lm_d.iteritems():
            lm_ids[i] = k
            lm_landmarks[i] = l.c_pointer
            i += 1

        lm_cptr = cls._call_cfunc(
            'vital_landmark_map_new',
            [t_lm_landmarks, t_lm_ids, ctypes.c_size_t],
            [lm_landmarks, lm_ids, s],
            cls.c_ptr_type()
        )
        return cls(lm_cptr)

    def __init__(self, from_cptr=None):
        """
        Create and empty map, or initialize from and existing C instance pointer
        :param from_cptr: Optional existing landmark map C pointer
        """
        super(LandmarkMap, self).__init__(from_cptr)

    def _new(self):
        return self._call_cfunc(
            'vital_landmark_map_new_empty',
            restype=self.C_TYPE_PTR
        )

    def _destroy(self):
        self._call_cfunc(
            'vital_landmark_map_destroy', [self.C_TYPE_PTR], [self]
        )

    def __eq__(self, other):
        return (
            isinstance(other, LandmarkMap) and
            self.as_dict() == other.as_dict()
        )

    def __ne__(self, other):
        return not (self == other)

    def __len__(self):
        return self.size

    @property
    def size(self):
        """
        Get the size of this map

        :return: the size of this map
        :rtype: int

        """
        return self._call_cfunc(
            'vital_landmark_map_size',
            [self.C_TYPE_PTR], [self],
            ctypes.c_size_t
        )

    def as_dict(self):
        """
        Get a copy of this map as a python dictionary

        :return: Dictionary mapping landmark IDs to Landmark instances
        :rtype: dict[int|long, vital.types.Landmark]

        """

        t_lm_ids = ctypes.POINTER(ctypes.c_int64)
        t_lm_landmarks = ctypes.POINTER(Landmark.c_ptr_type())

        lm_ids = t_lm_ids()
        lm_landmarks = t_lm_landmarks()

        self._call_cfunc(
            'vital_landmark_map_landmarks',
            [self.C_TYPE_PTR, ctypes.POINTER(t_lm_ids), ctypes.POINTER(t_lm_landmarks)],
            [self, ctypes.byref(lm_ids), ctypes.byref(lm_landmarks)]
        )

        d = {}
        s = self.size
        for i in xrange(s):
            # Need to copy ctypes pointer object
            l_cptr = Landmark.c_ptr_type()(lm_landmarks[i].contents)
            d[lm_ids[i]] = Landmark(from_cptr=l_cptr)

        free_void_ptr(lm_ids)
        free_void_ptr(lm_landmarks)

        return d
