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

Utility classes for wrapping locally-owned C pointer arrays as numpy arrays,
controlling destruction using python's reference counting.

"""
import numpy


class CptrWrapper (object):

    def __init__(self, cptr, free_func):
        self.cptr = cptr
        self.free = free_func

    def __del__(self):
        if self.free:
            self.free(self.cptr)


class CArrayWrapper (numpy.ndarray):

    def __new__(cls, cptr, size, ctype, free_func=None):
        """
        Wrap the given C data pointer as a numpy array with an optional data
        free function to be called when we're not referenced any more.

        :param cptr: ctypes C data array pointer
        :type cptr: _ctypes._Pointer

        :param size: Size of the C data array
        :type size: int

        :param ctype: C data type
        :type ctype: _ctypes._SimpleCData

        :param free_func: Optional function used to free the C data pointer when
            out of references. This function should take a single argument that
            is the ctypes pointer given to `cptr`.
        :type free_func: (_ctypes._Pointer) -> None

        """
        dtype = numpy.dtype(ctype)
        b = numpy.ctypeslib.as_array(cptr, (size,))
        obj = numpy.ndarray.__new__(cls, (size,), dtype, b)
        obj._cptr_container = CptrWrapper(cptr, free_func)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # Pass on c-pointer wrapping object if applicable
        self._cptr_container = getattr(obj, '_cptr_container', None)

    def __array_prepare__(self, obj, context=None):
        return obj

    def __array_wrap__(self, out, context=None):
        return out
