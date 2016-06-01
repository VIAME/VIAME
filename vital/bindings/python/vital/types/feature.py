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

vital::feature class interface

"""
import ctypes

import numpy

from vital.exceptions.base import VitalNoTypeInfoException
from vital.util import VitalObject
from vital.types import (
    Covariance,
    EigenArray,
    RGBColor,
)


class Feature (VitalObject):

    def __init__(self, loc=(0, 0), mag=0, scale=1, angle=0, rgb_color=None,
                 ctype=ctypes.c_double, from_cptr=None):
        """
        Create a new Feature instance.

        If created from an existing C reference, we inherently don't know the
        underlying data type (generic shared pointers under the hood). Thus,
        when creating a new Feature object from a C-pointer, the setter function
        cannon be used and instead raises a VitalNoTypeInfoException.

        :param loc: Location of the feature
        :param mag: Magnitude of the feature
        :param scale: Scale of the feature
        :param angle: Angle of the feature
        :param rgb_color: Color of the content under the feature
        :param ctype: Data type of the feature
        :param from_cptr: An existing feature instance to wrap.

        """
        if from_cptr is None:
            self._datatype = ctype
            # noinspection PyProtectedMember
            self._tchar = ctype._type_
        else:
            self._datatype = self._tchar = None
        super(Feature, self).__init__(from_cptr, loc, mag, scale, angle,
                                      rgb_color)

        # Test type integrity if given an explicit c-ptr (get type-name)
        # TODO

    def _new(self, loc, mag, scale, angle, rgb_color):
        loc = EigenArray.from_iterable(loc, target_ctype=self._datatype,
                                       target_shape=(2, 1))
        if rgb_color is None:
            rgb_color = RGBColor()

        # noinspection PyProtectedMember
        return self._call_cfunc(
            'vital_feature_{}_new'.format(self._tchar),
            [
                EigenArray.c_ptr_type(2, 1, self._datatype),
                self._datatype, self._datatype, self._datatype,
                RGBColor.c_ptr_type()
            ],
            [loc, mag, scale, angle, rgb_color],
            self.C_TYPE_PTR
        )

    def _destroy(self):
        self._call_cfunc(
            'vital_feature_destroy', [self.C_TYPE_PTR], [self]
        )

    def __eq__(self, other):
        if isinstance(other, Feature):
            return (
                numpy.array_equal(self.location, other.location) and
                self.magnitude == other.magnitude and
                self.scale == other.scale and
                self.angle == other.angle and
                self.covariance == other.covariance and
                self.color == other.color
            )
        return False

    def __ne__(self, other):
        return not (self == other)

    @property
    def type_name(self):
        """
        :return: Get the data type flag.
        :rtype: str
        """
        if not self._tchar:
            raise VitalNoTypeInfoException("Type info required but not present")
        return self._call_cfunc(
            'vital_feature_{}_type_name'.format(self._tchar),
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_char_p
        )

    @property
    def location(self):
        """ Get feature location """
        cptr = self._call_cfunc(
            'vital_feature_loc',
            [self.C_TYPE_PTR],
            [self],
            EigenArray.c_ptr_type(2),
        )
        return EigenArray(2, from_cptr=cptr)

    @location.setter
    def location(self, loc):
        """ Set new feature location
        :param loc: New locations. May be any iterable of 2 elements.
        :type loc: collections.Iterable
        """
        if self._datatype is None:
            raise VitalNoTypeInfoException("Type info required but not present")
        loc = EigenArray.from_iterable(loc, self._datatype, (2, 1))
        self._call_cfunc(
            'vital_feature_{}_set_loc'.format(self._tchar),
            [self.C_TYPE_PTR, loc.C_TYPE_PTR],
            [self, loc],
        )

    @property
    def magnitude(self):
        return self._call_cfunc(
            'vital_feature_magnitude',
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_double,
        )

    @magnitude.setter
    def magnitude(self, mag):
        if self._datatype is None:
            raise VitalNoTypeInfoException("Type info required but not present")
        self._call_cfunc(
            'vital_feature_{}_set_magnitude'.format(self._tchar),
            [self.C_TYPE_PTR, self._datatype],
            [self, mag],
        )

    @property
    def scale(self):
        return self._call_cfunc(
            'vital_feature_scale',
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_double,
        )

    @scale.setter
    def scale(self, scale):
        if self._datatype is None:
            raise VitalNoTypeInfoException("Type info required but not present")
        self._call_cfunc(
            'vital_feature_{}_set_scale'.format(self._tchar),
            [self.C_TYPE_PTR, self._datatype],
            [self, scale],
            None,
        )

    @property
    def angle(self):
        return self._call_cfunc(
            'vital_feature_angle',
            [self.C_TYPE_PTR],
            [self],
            ctypes.c_double,
        )

    @angle.setter
    def angle(self, angle):
        if self._datatype is None:
            raise VitalNoTypeInfoException("Type info required but not present")
        self._call_cfunc(
            'vital_feature_{}_set_angle'.format(self._tchar),
            [self.C_TYPE_PTR, self._datatype],
            [self, angle],
        )

    @property
    def covariance(self):
        cptr = self._call_cfunc(
            "vital_feature_covar",
            [self.C_TYPE_PTR],
            [self],
            Covariance.c_ptr_type(2, ctypes.c_double),
        )
        return Covariance(2, ctypes.c_double, from_cptr=cptr)

    @covariance.setter
    def covariance(self, covar):
        if self._datatype is None:
            raise VitalNoTypeInfoException("Type info required but not present")
        if not isinstance(covar, Covariance):
            # Try an make a covariance out of whatever was provided
            covar = Covariance(2, self._datatype, covar)
        print "Setting covar:", covar
        self._call_cfunc(
            "vital_feature_{}_set_covar".format(self._tchar),
            [self.C_TYPE_PTR, Covariance.c_ptr_type(2, self._datatype)],
            [self, covar],
        )

    @property
    def color(self):
        cptr = self._call_cfunc(
            "vital_feature_color",
            [self.C_TYPE_PTR],
            [self],
            RGBColor.c_ptr_type(),
        )
        return RGBColor(from_cptr=cptr)

    @color.setter
    def color(self, c):
        if self._datatype is None:
            raise VitalNoTypeInfoException("Type info required but not present")
        self._call_cfunc(
            'vital_feature_{}_set_color'.format(self._tchar),
            [self.C_TYPE_PTR, RGBColor.c_ptr_type()],
            [self, c],
        )
