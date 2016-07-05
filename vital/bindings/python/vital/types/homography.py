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

Interface to vital::homography

"""
import ctypes

import numpy

from vital.exceptions.math import PointMapsToInfinityException
from vital.types import (
    EigenArray
)
from vital.util import VitalObject, TYPE_NAME_MAP


class Homography (VitalObject):

    @classmethod
    def from_matrix(cls, m, datatype=ctypes.c_double):
        """
        Create a homography from an existing 3x3 matrix.

        If the data type of the matrix given, if an EigenArray, is not the same
        as ``datatype``, it will be automatically converted.

        :param m: Matrix to base the new homography on. This should be a 3x3
            matrix.
        :type m: collections.Iterable[collections.Iterable[float]] | vital.types.EigenArray

        :param datatype: Type to store data in the homography.
        :type datatype: ctypes._SimpleCData

        :return: New homography instance whose transform is equal to the given
            matrix.
        :rtype: Homography

        """
        # noinspection PyProtectedMember
        tchar = datatype._type_
        m = EigenArray.from_iterable(m, datatype, (3, 3))
        cptr = cls._call_cfunc(
            'vital_homography_%s_new_from_matrix' % tchar,
            [EigenArray.c_ptr_type(3, 3, datatype)], [m],
            Homography.c_ptr_type()
        )
        return Homography(from_cptr=cptr)

    def __init__(self, datatype=ctypes.c_double, from_cptr=None):
        """
        Create a new identity Homography instance or from an existing C pointer.

        :param datatype: The data type to represent homography as.
        :param from_cptr: An existing homography C instance pointer to wrap.

        """
        if from_cptr is None:
            # Initialize before entering _new
            self._datatype = datatype
            # noinspection PyProtectedMember
            self._tchar = datatype._type_

        super(Homography, self).__init__(from_cptr)

        # When constructing from an existing C-ptr, select datatype/tchar based
        # on self.type_name value
        if from_cptr is not None:
            self._datatype = TYPE_NAME_MAP[self.type_name]
            # noinspection PyProtectedMember
            self._tchar = self._datatype._type_

    def _new(self):
        return self._call_cfunc(
            'vital_homography_%s_new_identity' % self._tchar,
            restype=self.C_TYPE_PTR
        )

    def _destroy(self):
        self._call_cfunc('vital_homography_destroy', [self.C_TYPE_PTR], [self])

    def __mul__(self, other):
        if isinstance(other, Homography):
            h_cptr = self._call_cfunc(
                'vital_homography_%s_multiply' % self._tchar,
                [self.C_TYPE_PTR, self.C_TYPE_PTR],
                [self, other],
                self.C_TYPE_PTR
            )
            return Homography(from_cptr=h_cptr)
        else:
            raise ValueError("Not given a homography instance as the rhs "
                             "argument. Given %s type instead."
                             % type(other))

    def __eq__(self, other):
        """
        Test homography equality (matrix transform equality)
        :param other: Other object to check against
        :return: True of approximately equal transformation homographies
        """
        if isinstance(other, Homography):
            return numpy.allclose(self.as_matrix(), other.as_matrix())
        return False

    def __ne__(self, other):
        return not (self == other)

    @property
    def type_name(self):
        """
        :return: Get the data type name/flag.
        :rtype: str
        """
        return self._call_cfunc(
            'vital_homography_type_name', [self.C_TYPE_PTR], [self],
            ctypes.c_char_p
        )

    def clone(self):
        """
        :return: A new Homography instance that is a clone of this one
        :rtype: Homography
        """
        cptr = self._call_cfunc(
            'vital_homography_clone', [self.C_TYPE_PTR], [self], self.C_TYPE_PTR
        )
        return Homography(from_cptr=cptr)

    def normalize(self):
        """
        :return: A new Homography instance that is the normalized version of
            this.
        :rtype: Homography
        """
        cptr = self._call_cfunc(
            'vital_homography_normalize', [self.C_TYPE_PTR], [self],
            self.C_TYPE_PTR
        )
        return Homography(from_cptr=cptr)

    def inverse(self):
        """
        :return: A new Homography instance that is the inverse of this.
        :rtype: Homography
        """
        cptr = self._call_cfunc(
            'vital_homography_inverse', [self.C_TYPE_PTR], [self],
            self.C_TYPE_PTR
        )
        return Homography(from_cptr=cptr)

    def as_matrix(self):
        """
        :return: Get this homography as a 3x3 matrix (same data type as this
            homography).
        :rtype: EigenArray
        """
        m_cptr = self._call_cfunc(
            'vital_homography_%s_as_matrix' % self._tchar,
            [self.C_TYPE_PTR],
            [self],
            EigenArray.c_ptr_type(3, 3, self._datatype)
        )
        return EigenArray(3, 3, dtype=self._datatype, from_cptr=m_cptr)

    def map(self, point):
        """
        Map a 3D point using this homography

        If the data type of the point given, if an EigenArray, is not the same
        as this homography's ``datatype``, it will be automatically converted.

        :param point: 2D point to transform
        :type point: collections.Iterable[float] | EigenArray

        :return: Transformed 2D point as an EigenArray
        :rtype: EigenArray

        """
        point_t = EigenArray.c_ptr_type(2, ctype=self._datatype)
        point = EigenArray.from_iterable(point, self._datatype, (2, 1))
        p_cptr = self._call_cfunc(
            'vital_homography_%s_map_point' % self._tchar,
            [self.C_TYPE_PTR, point_t],
            [self, point],
            point_t,
            {
                1: PointMapsToInfinityException
            }
        )
        return EigenArray(2, dtype=self._datatype, from_cptr=p_cptr)
