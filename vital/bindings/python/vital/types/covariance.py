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

Interface to vital::covariance class

"""
import collections
import ctypes

import numpy

from vital.types.eigen import EigenArray
from vital.util import OpaqueTypeCache, VitalObject, VitalErrorHandle


class Covariance (VitalObject):

    # Override C opaque pointer type to ones that are dependent on size and type
    TYPE_CACHE = OpaqueTypeCache("Covariance_")
    C_TYPE = TYPE_CACHE.new_type_getter()
    C_TYPE_PTR = TYPE_CACHE.new_ptr_getter()

    SHAPE_SPEC = "{size:d}{type:s}"

    @classmethod
    def c_type(cls, size, ctype):
        """ Get the C opaque type """
        # noinspection PyProtectedMember
        return cls.C_TYPE[cls.SHAPE_SPEC.format(size=size, type=ctype._type_)]

    @classmethod
    def c_ptr_type(cls, N=2, ctype=ctypes.c_double):
        """ Get the C opaque pointer type """
        # noinspection PyProtectedMember
        return cls.C_TYPE_PTR[cls.SHAPE_SPEC.format(size=N,
                                                    type=ctype._type_)]

    def __init__(self, N=2, c_type=ctypes.c_double, init_scalar_or_matrix=None,
                 from_cptr=None):
        """
        Create a new Covariance symmetric matrix instance.

        This object stores the upper triangle portion of a symmetric matrix of
        side-length `N`. Accessing the lower triangle portion of the matrix thus
        yields the same values as the upper portion.

        :param N: Size of the matrix. This determines the side length of the
            symmetric matrix. This is constrained to the sizes declared in C:
            [2, 3]. Default is 2.
        :param c_type: The C data type to represent values in. This may me
            either float or double. Default is double.
        :param init_scalar_or_matrix: By default, we initialize an identity
            matrix. If a scalar value is provided here, we initialize to an
            identity times the given scalar. If it is a square EigenArray
            of size N, we initialize the covariance matrix based on this matrix
            (averages off diagonal elements to enforce symmetry). Input matrix
            data is copied, not shared.
        :param from_cptr: Existing C opaque instance pointer to use, preventing new
            instance construction. This should of course be a valid pointer to
            an instance.

        """
        # Initialize C type and pointer + function map
        # noinspection PyProtectedMember
        c_type_char = c_type._type_
        ss = self.SHAPE_SPEC.format(size=N, type=c_type_char)

        # Set concrete shape/type specific opaque pointer for this instance
        self.C_TYPE = self.__class__.C_TYPE[ss]
        self.C_TYPE_PTR = self.__class__.C_TYPE_PTR[ss]

        # Initialize function map based on shape
        self._func_map = {
            'new_identity':
                self.VITAL_LIB['vital_covariance_{}_new'.format(ss)],
            'new_scalar':
                self.VITAL_LIB['vital_covariance_{}_new_from_scalar'.format(ss)],
            'new_matrix':
                self.VITAL_LIB['vital_covariance_{}_new_from_matrix'.format(ss)],
            'destroy':
                self.VITAL_LIB['vital_covariance_{}_destroy'.format(ss)],
            'to_matrix':
                self.VITAL_LIB['vital_covariance_{}_to_matrix'.format(ss)],
            'get':
                self.VITAL_LIB['vital_covariance_{}_get'.format(ss)],
            'set':
                self.VITAL_LIB['vital_covariance_{}_set'.format(ss)],
        }
        self._N = N
        self._ctype = c_type

        # Now that we have a concrete opaque struct types...
        super(Covariance, self).__init__(from_cptr, init_scalar_or_matrix)

    def _new(self, init_scalar_or_matrix):
        """
        Construct a new instance, returning new instance opaque C pointer and
        initializing any other necessary object properties

        :returns: New C opaque structure pointer.

        """
        N = self._N
        c_type = self._ctype

        # Choose relevant constructor
        if init_scalar_or_matrix is None:
            self._log.debug("Initializing identity")
            c_new = self._func_map['new_identity']
            c_new.argtypes = [VitalErrorHandle.C_TYPE_PTR]
            c_new.restype = self.C_TYPE_PTR
            args = ()
        elif isinstance(init_scalar_or_matrix, (collections.Iterable,
                                                numpy.ndarray)):
            self._log.debug("Initializing with matrix")
            # Should be a NxN square matrix
            mat = EigenArray.from_iterable(init_scalar_or_matrix, c_type,
                                           (N, N))
            c_new = self._func_map['new_matrix']
            c_new.argtypes = [mat.C_TYPE_PTR,
                              VitalErrorHandle.C_TYPE_PTR]
            args = (mat,)
        else:
            self._log.debug("Initializing with scalar")
            c_new = self._func_map['new_scalar']
            c_new.argtypes = [self._ctype, VitalErrorHandle.C_TYPE_PTR]
            args = (init_scalar_or_matrix,)

        c_new.restype = self.C_TYPE_PTR
        with VitalErrorHandle() as eh:
            c_args = args + (eh,)
            # self._log.debug("Construction args: %s", c_args)
            c_ptr = c_new(*c_args)
        if not bool(c_ptr):
            raise RuntimeError("C++ Construction failed (null pointer)")
        return c_ptr

    def _destroy(self):
        c_del = self._func_map['destroy']
        c_del.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            c_del(self, eh)
        self._inst_ptr = self.C_TYPE_PTR()

    def to_matrix(self):
        c_to_mat = self._func_map['to_matrix']
        c_to_mat.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        c_to_mat.restype = EigenArray.c_ptr_type(self._N, self._N, self._ctype)

        with VitalErrorHandle() as eh:
            m_ptr = c_to_mat(self, eh)
            return EigenArray(self._N, self._N, dtype=numpy.dtype(self._ctype),
                              from_cptr=m_ptr, owns_data=True)

    def __repr__(self):
        return "%s{\n%s}" % (self.__class__.__name__, self.to_matrix())

    def __str__(self):
        return str(self.to_matrix())

    def __eq__(self, other):
        if isinstance(other, Covariance):
            return numpy.allclose(self.to_matrix(), other.to_matrix())
        return False

    def __ne__(self, other):
        return not (self == other)

    def __getitem__(self, p):
        """
        Get an element of the covariance matrix
        :param p: row, column index pair
        :return: Value at the specified index in if bounds
        :raises IndexError: Index out of bounds
        """
        r, c = p
        if not (0 <= r < self._N and 0 <= c < self._N):
            raise IndexError(p)
        c_get = self._func_map['get']
        c_get.argtypes = [self.C_TYPE_PTR, ctypes.c_uint, ctypes.c_uint,
                          VitalErrorHandle.C_TYPE_PTR]
        c_get.restype = self._ctype
        with VitalErrorHandle() as eh:
            return c_get(self, r, c, eh)

    def __setitem__(self, p, value):
        r, c = p
        if not (0 <= r < self._N and 0 <= c < self._N):
            raise IndexError(p)
        c_set = self._func_map['set']
        c_set.argtypes = [self.C_TYPE_PTR, ctypes.c_uint, ctypes.c_uint,
                          self._ctype, VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            c_set(self, r, c, value, eh)
