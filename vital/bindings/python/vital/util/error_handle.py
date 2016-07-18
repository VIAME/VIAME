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

Vital Python Error handler class

"""
# -*- coding: utf-8 -*-
__author__ = 'paul.tunison@kitware.com'

import ctypes

from vital.util.VitalObject import VitalObject
from vital.exceptions.base import VitalBaseException


# noinspection PyPep8Naming
class VitalErrorHandle (VitalObject):
    """ Error handling structure used in C interface """

    # noinspection PyPep8Naming
    class C_TYPE (ctypes.Structure):
        """
        C Interface structure
        """
        _fields_ = [
            ("error_code", ctypes.c_int),
            ("message", ctypes.c_char_p),
        ]

    C_TYPE_PTR = ctypes.POINTER(C_TYPE)

    def __init__(self):
        super(VitalErrorHandle, self).__init__()
        self._ec_exception_map = {}

    def _new(self):
        """
        Create a new error handle instance.
        """
        eh_new = self.VITAL_LIB['vital_eh_new']
        eh_new.restype = self.C_TYPE_PTR
        c_ptr = eh_new()
        if not c_ptr:
            raise RuntimeError("Failed construct new error handle instance")
        return c_ptr

    def _destroy(self):
        eh_del = self.VITAL_LIB['vital_eh_destroy']
        eh_del.argtypes = [self.C_TYPE_PTR]
        eh_del(self._inst_ptr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        else:
            self.propagate_exception()
        return True

    @property
    def error_code(self):
        return self.c_pointer[0].error_code

    @property
    def message(self):
        return self.c_pointer[0].message

    def set_exception_map(self, ec_exception_map):
        """
        Extend the current return code to exception mapping.

        :param ec_exception_map: Dictionary mapping integer return code to an
            exception, or function returning an exception instance, that should
            be raised.
        :type ec_exception_map: dict[int, BaseException | types.FunctionType]

        """
        self._ec_exception_map.update(ec_exception_map)

    def propagate_exception(self):
        """
        Raise appropriate Python exception if our current error code is non-zero

        By default, if a non-zero error code is observed, a generic
        VitalBaseException is raised with the provided error handle message.

        If an exception map was set via set_exception_map(...) and the error
        code matches an entry, that will be raised instead.

        """
        if self.error_code != 0:
            if self.error_code in self._ec_exception_map:
                raise self._ec_exception_map[self.error_code](self.message)
            else:
                raise VitalBaseException(self.message)
