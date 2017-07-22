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

Vital landmark class interface

"""
import ctypes

import numpy

from vital.exceptions.base import VitalDynamicCastException
from vital.types import Covariance
from vital.types import EigenArray
from vital.types import RGBColor
from vital.util import VitalObject, TYPE_NAME_MAP


class Landmark (VitalObject):
    """
    Landmark class interface

    Property accessors return double-types objects, but setting properties
    required objects with the same underlying type as this instance (usually
    converted if given otherwise).

    """

    def __init__(self, loc=(0, 0, 0), scale=1, c_type=ctypes.c_double,
                 from_cptr=None):
        if from_cptr is None:
            # Initialize type and char before entering _new
            self._datatype = c_type
            # noinspection PyProtectedMember
            self._tchar = self._datatype._type_

        super(Landmark, self).__init__(from_cptr, loc, scale)

        # When initializing from a c-ptr, _new isn't called and _datatype and
        # _tchar are not initialized. So introspect them here.
        if from_cptr is not None:
            self._datatype = TYPE_NAME_MAP[self.type_name]
            # noinspection PyProtectedMember
            self._tchar = self._datatype._type_

    def _new(self, loc, scale):
        loc = EigenArray.from_iterable(loc, self._datatype, (3, 1))
        return self._call_cfunc(
            'vital_landmark_{}_new_scale'.format(self._tchar),
            [EigenArray.c_ptr_type(3, ctype=self._datatype), self._datatype],
            [loc, scale],
            self.C_TYPE_PTR
        )

    def _destroy(self):
        self._call_cfunc('vital_landmark_destroy', [self.C_TYPE_PTR], [self])

    def __eq__(self, other):
        if isinstance(other, Landmark):
            return all((
                numpy.allclose(self.loc, other.loc),
                self.scale == other.scale,
                numpy.allclose(self.normal, other.normal),
                self.covariance == other.covariance,
                self.color == other.color,
                self.observations == other.observations,
            ))
        return False

    def __ne__(self, other):
        return not (self == other)

    def clone(self):
        """
        :return: Return a new instance that is the clone of this one.
        :rtype: Landmark
        """
        cptr = self._call_cfunc(
            'vital_landmark_clone',
            [self.C_TYPE_PTR], [self],
            self.C_TYPE_PTR
        )
        return Landmark(from_cptr=cptr)

    @property
    def type_name(self):
        return self._call_cfunc(
            'vital_landmark_type_name',
            [self.C_TYPE_PTR], [self],
            ctypes.c_char_p
        )

    @property
    def datatype(self):
        """
        Ctypes type the underlying data is stored as.
        :rtype: ctypes._SimpleCData
        """
        return self._datatype

    @property
    def loc(self):
        """
        Get the 3D location of this landmark
        :return: 3D location of this landmark
        :rtype: EigenArray
        """
        cptr = self._call_cfunc(
            'vital_landmark_loc',
            [self.C_TYPE_PTR], [self],
            EigenArray.c_ptr_type(3)
        )
        return EigenArray(3, from_cptr=cptr)

    @loc.setter
    def loc(self, new_loc):
        """
        Set the 3D location of this landmark
        :param new_loc: New 3D location
        :type new_loc: collections.Iterable[float]
        """
        new_loc = EigenArray.from_iterable(new_loc, self._datatype, (3, 1))
        self._call_cfunc(
            'vital_landmark_{}_set_loc'.format(self._tchar),
            [self.C_TYPE_PTR, EigenArray.c_ptr_type(3, ctype=self._datatype)],
            [self, new_loc],
            exception_map={
                1: VitalDynamicCastException
            }
        )

    @property
    def scale(self):
        return self._call_cfunc(
            'vital_landmark_scale',
            [self.C_TYPE_PTR], [self],
            ctypes.c_double
        )

    @scale.setter
    def scale(self, s):
        self._call_cfunc(
            'vital_landmark_{}_set_scale'.format(self._tchar),
            [self.C_TYPE_PTR, self._datatype],
            [self, s],
            exception_map={
                1: VitalDynamicCastException
            }
        )

    @property
    def normal(self):
        cptr = self._call_cfunc(
            'vital_landmark_normal',
            [self.C_TYPE_PTR], [self],
            EigenArray.c_ptr_type(3)
        )
        return EigenArray(3, from_cptr=cptr)

    @normal.setter
    def normal(self, n):
        n = EigenArray.from_iterable(n, self._datatype, (3, 1))
        self._call_cfunc(
            'vital_landmark_{}_set_normal'.format(self._tchar),
            [self.C_TYPE_PTR, EigenArray.c_ptr_type(3, ctype=self._datatype)],
            [self, n],
            exception_map={
                1: VitalDynamicCastException
            }
        )

    @property
    def covariance(self):
        cptr = self._call_cfunc(
            'vital_landmark_covariance',
            [self.C_TYPE_PTR], [self],
            Covariance.c_ptr_type(3)
        )
        return Covariance(3, from_cptr=cptr)

    @covariance.setter
    def covariance(self, c):
        """
        Set new landmark covariance
        :type c: vital.types.Covariance
        """
        expected_covar_type = Covariance.c_ptr_type(3, self._datatype)
        # Convert covariance to correct type if necessary
        if c.C_TYPE_PTR != expected_covar_type:
            c = Covariance(3, self._datatype, c.to_matrix())
        self._call_cfunc(
            'vital_landmark_{}_set_covar'.format(self._tchar),
            [self.C_TYPE_PTR, expected_covar_type],
            [self, c],
            exception_map={
                1: VitalDynamicCastException
            }
        )

    @property
    def color(self):
        cptr = self._call_cfunc(
            'vital_landmark_color',
            [self.C_TYPE_PTR], [self],
            RGBColor.c_ptr_type()
        )
        return RGBColor(from_cptr=cptr)

    @color.setter
    def color(self, c):
        self._call_cfunc(
            'vital_landmark_{}_set_color'.format(self._tchar),
            [self.C_TYPE_PTR, RGBColor.c_ptr_type()],
            [self, c],
            exception_map={
                1: VitalDynamicCastException
            }
        )

    @property
    def observations(self):
        return self._call_cfunc(
            'vital_landmark_observations',
            [self.C_TYPE_PTR], [self],
            ctypes.c_uint
        )

    @observations.setter
    def observations(self, o):
        if o < 0:
            raise ValueError("Observations just be an unsigned integer")

        self._call_cfunc(
            'vital_landmark_{}_set_observations'.format(self._tchar),
            [self.C_TYPE_PTR, ctypes.c_uint],
            [self, o],
            exception_map={
                1: VitalDynamicCastException
            }
        )
