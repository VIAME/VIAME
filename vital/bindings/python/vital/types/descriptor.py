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

vital::descriptor interface

"""
import ctypes

import numpy

from vital.exceptions.base import VitalDynamicCastException
from vital.util import (
    free_void_ptr,
    TYPE_NAME_MAP,
    VitalErrorHandle,
    VitalObject,
)
from vital.util.array_wrapping import CArrayWrapper


class Descriptor (numpy.ndarray, VitalObject):
    """
    Vital descriptor class interface class
    """
    # Only has one C type and pointer type due to shared nature in C++, but
    # retains knowing what type initialized with for accessing type-specific
    # functions.

    def __new__(cls, size=128, ctype=ctypes.c_double, from_cptr=None):
        """
        Create a descriptor instance

        :param size: Size of the descriptor (number of elements). Default of 128
            (arbitrary).
        :param ctype: Data type that this data is represented as under the hood.
        :param from_cptr: Existing Descriptor instance to wrap.
        """
        if from_cptr is None:
            d_type = ctype
            # noinspection PyProtectedMember
            d_type_char = ctype._type_
            d_new = cls.VITAL_LIB['vital_descriptor_new_{}'.format(d_type_char)]
            d_new.argtypes = [ctypes.c_size_t, VitalErrorHandle.c_ptr_type()]
            d_new.restype = cls.c_ptr_type()
            with VitalErrorHandle() as eh:
                inst_ptr = d_new(size, eh)
        else:
            if not isinstance(from_cptr, cls.c_ptr_type()):
                raise ValueError("Invalid ``from_cptr`` value (given %s"
                                 % type(from_cptr))
            inst_ptr = from_cptr
            # Get type char from generic data type introspection function
            # ASSUMING typename from c++ is the same as ctypes _type_ values,
            #   which is at least currently true for float/double types, which
            #   is all that we care about / is implemented in C/C++.
            d_type = TYPE_NAME_MAP[cls._call_cfunc(
                'vital_descriptor_type_name',
                [cls.c_ptr_type()],
                [inst_ptr],
                ctypes.c_char_p
            )]
            # noinspection PyProtectedMember
            d_type_char = d_type._type_
            # Extract existing instance size information
            size = cls._call_cfunc(
                'vital_descriptor_size',
                [cls.c_ptr_type()],
                [inst_ptr],
                ctypes.c_size_t
            )

        # Get the raw-data pointer from inst to wrap array around
        d_raw_data = cls.VITAL_LIB['vital_descriptor_get_{}_raw_data'
                                   .format(d_type_char)]
        d_raw_data.argtypes = [cls.c_ptr_type(), VitalErrorHandle.c_ptr_type()]
        d_raw_data.restype = ctypes.POINTER(d_type)
        # TODO: We could recover from an exception here by parsing the type
        #       expected in the error message and changing the construction type
        with VitalErrorHandle() as eh:
            eh.set_exception_map({1: VitalDynamicCastException})
            data_ptr = d_raw_data(inst_ptr, eh)
        b = numpy.ctypeslib.as_array(data_ptr, (size,))

        npy_type = numpy.dtype(d_type)
        obj = numpy.ndarray.__new__(cls, size, npy_type, b)
        obj._inst_ptr = inst_ptr
        obj._owns_data = True  # This is the owning instance
        return obj

    def __init__(self, size=128, ctype=ctypes.c_double, from_cptr=None):
        # __new__ creates or sets _inst_ptr, so always supply that to super's
        # `from_cptr`'
        VitalObject.__init__(self, self._inst_ptr)

    __init__.__doc__ = __new__.__doc__

    def __array_finalize__(self, obj):
        if obj is None:  # When being constructed
            return

        # View-casting or creating from template
        if isinstance(obj, Descriptor):
            self._inst_ptr = obj._inst_ptr
            self._owns_data = False  # Views/children don't own data instance

    def __array_prepare__(self, obj, context=None):
        return obj

    def __array_wrap__(self, out, context=None):
        return out

    def _destroy(self):
        if hasattr(self, '_owns_data') and self._owns_data:
            d_del = self.VITAL_LIB['vital_descriptor_destroy']
            d_del.argtypes = [self.c_ptr_type(), VitalErrorHandle.c_ptr_type()]
            with VitalErrorHandle() as eh:
                d_del(self, eh)

    def tobytearray(self):
        """
        :return: a copy of the descriptor vector as a numpy.ndarray. Copied data
            will be of bytes regardless of stored data type.
        :rtype: vital.util.array_wrapping.CArrayWrapper
        """
        d_as_bytes = self.VITAL_LIB['vital_descriptor_as_bytes']
        d_as_bytes.argtypes = [self.c_ptr_type(), VitalErrorHandle.c_ptr_type()]
        d_as_bytes.restype = ctypes.POINTER(ctypes.c_uint8)

        with VitalErrorHandle() as eh:
            cptr = d_as_bytes(self, eh)

        return CArrayWrapper(cptr, self.nbytes, ctypes.c_uint8, free_void_ptr)
