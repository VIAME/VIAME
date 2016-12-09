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

Interface to vital::rgb_color

"""
import ctypes

import numpy

from vital.util import VitalObject


class RGBColor (VitalObject):
    """
    Pixel RGB color representation.

    While not a numpy array, this class can be used
    """

    def __init__(self, r=255, g=255, b=255, from_cptr=None):
        """
        Create a new color instance. Color values are cast to integers.

        :param r: Red color value
        :param g: Green color value
        :param b: Blue color value
        :param from_cptr: Existing C pointer to initialize to.

        """
        super(RGBColor, self).__init__(from_cptr, r, g, b)

    def _new(self, r, g, b):
        return self._call_cfunc(
            'vital_rgb_color_new',
            [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte],
            [int(r), int(g), int(b)],
            self.c_ptr_type(),
        )

    def _destroy(self):
        self._call_cfunc(
            "vital_rgb_color_destroy",
            [self.c_ptr_type()],
            [self],
        )

    def __getitem__(self, idx):
        # Using array for out-of-bounds error handling and other conveniences
        return self.__array__()[idx]

    def __array__(self, dtype=None):
        if dtype is None:
            dtype = numpy.ubyte
        return numpy.array([self.r, self.g, self.b], dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, RGBColor):
            return (
                self.r == other.r and
                self.g == other.g and
                self.b == other.b
            )

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s{%d, %d, %d}" % (self.__class__.__name__,
                                   self.r, self.g, self.b)

    @property
    def r(self):
        return self._call_cfunc(
            'vital_rgb_color_r',
            [self.c_ptr_type()],
            [self],
            ctypes.c_ubyte
        )

    @property
    def g(self):
        return self._call_cfunc(
            'vital_rgb_color_g',
            [self.c_ptr_type()],
            [self],
            ctypes.c_ubyte
        )

    @property
    def b(self):
        return self._call_cfunc(
            'vital_rgb_color_b',
            [self.c_ptr_type()],
            [self],
            ctypes.c_ubyte
        )
