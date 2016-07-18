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

Interface to VITAL Eigen matrix classes through numpy

"""
import ctypes

import numpy

from vital.exceptions.eigen import VitalInvalidStaticEigenShape
from vital.util import (
    VitalErrorHandle,
    VitalObject,
)
from vital.util.VitalObject import OpaqueTypeCache


__author__ = 'paul.tunison@kitware.com'


if ctypes.sizeof(ctypes.c_void_p) == 4:
    c_ptrdiff_t = ctypes.c_int32
elif ctypes.sizeof(ctypes.c_void_p) == 8:
    c_ptrdiff_t = ctypes.c_int64
else:
    raise RuntimeError("Invalid c_void_p size? =%d"
                       % ctypes.sizeof(ctypes.c_void_p))


class EigenArray (numpy.ndarray, VitalObject):
    """
    TODO: Figure out why ravel/flatten returns a VitalEigenArray-type object
          with no bases

    This class ignores the base C_TYPE and C_TYPE_POINTER in lue of a
    shape-dependent opaque C-pointer type that is initialized when encountering
    a specific shape type (substituting in dynamics for size where appropriate).

    """

    # Valid dtype possibilities
    MAT_TYPE_KEYS = (numpy.double, numpy.float32)

    # C library function component template
    FUNC_SPEC = "{rows:s}x{cols:s}{type:s}"

    # Override C opaque pointer type to ones that are dependent on shape and
    # dynamics.
    TYPE_CACHE = OpaqueTypeCache("EigenArray_")
    C_TYPE = TYPE_CACHE.new_type_getter()
    C_TYPE_PTR = TYPE_CACHE.new_ptr_getter()

    __array_priority__ = -1.0

    @classmethod
    def c_type(cls, rows, cols, ctype):
        """
        Helper class function to get the correct C opaque pointer type for a
        given shape and data type.

        :param rows: Number of static rows in matrix. 'X' for dynamic.
        :param cols: Number of static columns in matrix. 'X' for dynamic.
        :param ctype: The C data type the data is stored in (use ctypes)
        :return: C opaque structure type

        """
        # noinspection PyProtectedMember
        return cls.C_TYPE[cls.FUNC_SPEC.format(rows=str(rows),
                                               cols=str(cols),
                                               type=ctype._type_)]

    @classmethod
    def c_ptr_type(cls, rows, cols=1, ctype=ctypes.c_double):
        """
        Helper class function to get the correct C opaque pointer type for a
        given shape and data type.

        :param rows: Number of static rows in matrix. 'X' for dynamic.
        :param cols: Number of static columns in matrix. 'X' for dynamic.
        :param ctype: The C data type the data is stored in (use ctypes)
        :return: C opaque structure pointer type

        """
        # noinspection PyProtectedMember
        return cls.C_TYPE_PTR[cls.FUNC_SPEC.format(rows=str(rows),
                                                   cols=str(cols),
                                                   type=ctype._type_)]

    @classmethod
    def _init_func_map(cls, rows, cols, d_rows, d_cols, dtype):
        """
        Initialize C function naming map for the given shape and type
        information.

        :returns: Function name mapping and associated ctypes data type
        :rtype: (str, dict[str, str], _ctypes._SimpleCData)

        """
        if dtype == cls.MAT_TYPE_KEYS[0]:  # C double
            type_char = 'd'
            c_type = ctypes.c_double
        elif dtype == cls.MAT_TYPE_KEYS[1]:  # C float
            type_char = 'f'
            c_type = ctypes.c_float
        else:
            raise ValueError("Invalid data type given ('%s'). "
                             "Must be one of %s."
                             % (dtype, cls.MAT_TYPE_KEYS))
        func_spec = cls.FUNC_SPEC.format(
            rows=(d_rows and 'X') or str(rows),
            cols=(d_cols and 'X') or str(cols),
            type=type_char,
        )

        func_map = {
            'new': 'vital_eigen_matrix{}_new'.format(func_spec),
            'new_sized': 'vital_eigen_matrix{}_new_sized'.format(func_spec),
            'destroy': 'vital_eigen_matrix{}_destroy'.format(func_spec),
            'get': 'vital_eigen_matrix{}_get'.format(func_spec),
            'set': 'vital_eigen_matrix{}_set'.format(func_spec),
            'rows': 'vital_eigen_matrix{}_rows'.format(func_spec),
            'cols': 'vital_eigen_matrix{}_cols'.format(func_spec),
            'row_stride': 'vital_eigen_matrix{}_row_stride'.format(func_spec),
            'col_stride': 'vital_eigen_matrix{}_col_stride'.format(func_spec),
            'data': 'vital_eigen_matrix{}_data'.format(func_spec),
        }
        return func_spec, func_map, c_type

    @classmethod
    def _get_data_components(cls, ptr, ptr_type, c_type, func_map):
        """
        Get underlying Eigen matrix shape, stride and data pointer
        """
        v_rows = cls.VITAL_LIB[func_map['rows']]
        v_cols = cls.VITAL_LIB[func_map['cols']]
        v_row_stride = cls.VITAL_LIB[func_map['row_stride']]
        v_col_stride = cls.VITAL_LIB[func_map['col_stride']]
        v_data = cls.VITAL_LIB[func_map['data']]

        v_rows.argtypes = [ptr_type, VitalErrorHandle.C_TYPE_PTR]
        v_cols.argtypes = [ptr_type, VitalErrorHandle.C_TYPE_PTR]
        v_row_stride.argtypes = [ptr_type, VitalErrorHandle.C_TYPE_PTR]
        v_col_stride.argtypes = [ptr_type, VitalErrorHandle.C_TYPE_PTR]
        v_data.argtypes = [ptr_type, VitalErrorHandle.C_TYPE_PTR]

        v_rows.restype = c_ptrdiff_t
        v_cols.restype = c_ptrdiff_t
        v_row_stride.restype = c_ptrdiff_t
        v_col_stride.restype = c_ptrdiff_t
        v_data.restype = ctypes.POINTER(c_type)

        with VitalErrorHandle() as eh:
            rows = v_rows(ptr, eh)
            cols = v_cols(ptr, eh)
            row_stride = v_row_stride(ptr, eh)
            col_stride = v_col_stride(ptr, eh)
            data = v_data(ptr, eh)

        return rows, cols, row_stride, col_stride, data

    @classmethod
    def from_iterable(cls, i, target_ctype=ctypes.c_double, target_shape=None):
        """
        Try to create an EigenArray given an iterable of values.

        1-dimensional iterables are interpreted as column-vectors, unless `i` is
        an EigenArray, in which case it is returned as is.

        This function is limited to Eigen statically defined matrix shapes
        (dynamic rows / cols not used).

        If the input ``i`` is already an EigenArray, this function does nothing
        and the input array is returned.

        :param i: Source array-like data
        :type i: collections.Iterable | numpy.ndarray | EigenArray

        :param target_shape: The intended result array shape for error checking.
            If None, we will not assert a shape.
        :type target_shape: None | (int, int)

        :param target_ctype: Target result EigenArray data type
        :type target_ctype: _ctypes._SimpleCData

        :return: New EigenArray instance that is the same shape as the input
            data.
        :rtype: EigenArray

        """
        # Make input iterable into an actual numpy.ndarray if it wasn't already
        #: :type: numpy.ndarray | EigenArray
        vec = numpy.array(i, copy=False, subok=True)
        if vec.ndim == 1:
            vec = vec[:, numpy.newaxis]

        # Transform into EigenArray if not already
        if not isinstance(vec, EigenArray) or vec._c_type != target_ctype:
            tmp = EigenArray(*vec.shape, dtype=target_ctype)
            tmp[:] = vec
            vec = tmp

        if target_shape and vec.shape != target_shape:
            raise ValueError("Incorrect shape %s. Expecting %s."
                             % (str(vec.shape), str(target_shape)))

        return vec

    def __new__(cls, rows=2, cols=1, dynamic_rows=False, dynamic_cols=False,
                dtype=numpy.double, from_cptr=None, owns_data=True):
        """
        Create a new Vital Eigen matrix and interface

        :param rows: Number of rows in the matrix
        :param cols: Number of columns in the matrix
        :param dynamic_rows: If we should not use compile-time generated types
            in regards to the row specification.
        :param dynamic_cols: If we should not use compile-time generated types
            in regards to the column specification.
        :param dtype: numpy dtype to use
        :param from_cptr: Optional existing C Eigen matrix instance pointer to
            use instead of constructing a new one.
        :param owns_data: When given a c-pointer, if we should take ownership of
            the underlying data.

        :return: Interface to a new or existing Eigen matrix instance.

        :raises ValueError: If a `c_ptr` is provided but is of a different
            compile-time shape type. E.g. cannot create a 2x1 matrix with
            dynamic rows from a 2x1 matrix
        :raises VitalInvalidStaticEigenShape: An invalid (row, column) was
            specified without stating dynamic rows or columns. This is because
            Vital and Eigen defines only so many shapes at compile time.

        """
        dtype = numpy.dtype(dtype)
        func_spec, func_map, c_type = \
            cls._init_func_map(rows, cols, dynamic_rows, dynamic_cols, dtype)
        op_c_type, op_c_type_ptr = cls.TYPE_CACHE.get_types(func_spec)

        # Create new Eigen matrix
        # Check if the shape given is either fully dynamic or is a valid
        # compile-time shape
        try:
            cls.VITAL_LIB[func_map['new_sized']]
        except AttributeError:
            raise VitalInvalidStaticEigenShape(
                "Array shape %s is not a valid compile-time specified "
                "shape given that dynamic rows/cols were specified as %s."
                % ((rows, cols), (bool(dynamic_rows), bool(dynamic_cols)))
            )
        if from_cptr is None:
            c_new = cls.VITAL_LIB[func_map['new_sized']]
            c_new.argtypes = [c_ptrdiff_t, c_ptrdiff_t]
            c_new.restype = op_c_type_ptr
            inst_ptr = c_new(c_ptrdiff_t(rows), c_ptrdiff_t(cols))
            if not bool(inst_ptr):
                raise RuntimeError("Failed to construct new Eigen matrix")
            owns_data = True
        else:
            if not isinstance(from_cptr, op_c_type_ptr):
                raise ValueError("Given C-Pointer is not correct for the "
                                 "shape-type '%s' (given: %s)"
                                 % (func_spec, type(from_cptr)))
            inst_ptr = from_cptr

        # Get information, data pointer and base transformed array
        rows, cols, row_stride, col_stride, data = \
            cls._get_data_components(inst_ptr, op_c_type_ptr, c_type, func_map)
        # Might have to swap out the use of ``dtype_bytes`` for
        # inner/outer size values from Eigen if matrices are ever NOT
        # densely packed.
        dtype_bytes = numpy.dtype(c_type).alignment
        strides = (row_stride * dtype_bytes,
                   col_stride * dtype_bytes)
        if data:
            # data has no __array_interface__ yet, thus...
            b = numpy.ctypeslib.as_array(data, (rows * cols,))
        else:
            b = buffer('')

        # args: (subclass, shape, dtype, buffer, offset, strides, order)
        obj = numpy.ndarray.__new__(cls, (rows, cols), dtype, b, 0, strides)

        # local properties
        # instance-specific opaque type
        obj.C_TYPE = op_c_type
        obj.C_TYPE_PTR = op_c_type_ptr
        obj._dynamic_rows = dynamic_rows
        obj._dynamic_cols = dynamic_cols
        obj._func_map = func_map
        obj._c_type = c_type
        obj._owns_data = owns_data

        # Always going to have an instance pointer at this point due to above
        # logic
        VitalObject.__init__(obj, from_cptr=inst_ptr)

        return obj

    # noinspection PyMissingConstructor
    def __init__(self, rows=2, cols=1, dynamic_rows=False, dynamic_cols=False,
                 dtype=numpy.double, from_cptr=None, owns_data=True):
        """
        Create a new Vital Eigen matrix instance.

        :param rows: Number of rows in the matrix
        :param cols: Number of columns in the matrix
        :param dynamic_rows: If we should not use compile-time generated types
            in regards to the row specification.
        :param dynamic_cols: If we should not use compile-time generated types
            in regards to the column specification.
        :param dtype: numpy dtype to use
        :param from_cptr: Optional existing C Eigen matrix instance pointer to use
            instead of constructing a new one.
        :param owns_data: When given a c-pointer, if we should take ownership of
            the underlying data.

        :return: Interface to a new or existing Eigen matrix instance.

        :raises ValueError: If a `c_ptr` is provided but is of a different
            compile-time shape type. E.g. cannot create a 2x1 matrix with
            dynamic rows from a 2x1 matrix
        :raises VitalInvalidStaticEigenShape: An invalid (row, column) was
            specified without stating dynamic rows or columns. This is because
            Vital and Eigen defines only so many shapes at compile time.

        """
        # initialization handled in __new__
        # function args above required in order to construct

    def __repr__(self):
        cls_name = self.__class__.__name__
        s = str(self)
        # prefix lines based on the length of the class name
        l = s.splitlines()
        l[0] = cls_name + '(' + l[0]
        for i in range(1, len(l)):
            if l[i]:
                # +1 for the '('
                l[i] = ' '*(len(cls_name)+1) + l[i]
        l[-1] += ')'
        return '\n'.join(l)

    def __array_finalize__(self, obj):
        """
        Where numpy finalizes instance properties of an array instance when
        created due to __new__, casting or new-from-template.
        """
        # got here from __new__, nothing to transfer
        if obj is None:
            return

        # copy/move over attributes from parent as necessary
        #   self => New class of this type
        #   obj  => other class MAYBE this type
        if isinstance(obj, EigenArray):
            self._dynamic_rows = obj._dynamic_rows
            self._dynamic_cols = obj._dynamic_cols
            self._func_map = obj._func_map
            self._c_type = obj._c_type
            # Always false because we are view of obj. See parent switch below.
            self._owns_data = False

            self._inst_ptr = obj._inst_ptr
        else:
            raise RuntimeError("Finalizing VitalEigenNumpyArray whose parent "
                               "is not of the same type (%s). Cannot inherit "
                               "required information." % type(obj))

    def __array_prepare__(self, obj, context=None):
        # Don't propagate this class and its stored references needlessly
        # NOTE: could make a new EigenArray of the same shape (separate memory)
        #       here and return that.
        return obj

    def __array_wrap__(self, out_arr, context=None):
        # Don't propagate this class and its stored references needlessly
        return out_arr

    def _new(self):
        """
        Spoof method because we're descending from numpy.ndarray, which changes
        how construction occurs.
        """

    def _destroy(self):
        # We're only in this function because we don't have a parent.
        # Not smart-pointer controlled in C++. We might not own the data we're
        # viewing.
        if self.c_pointer and self._owns_data:
            # print "Destroying"
            m_del = self.VITAL_LIB[self._func_map['destroy']]
            m_del.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
            with VitalErrorHandle() as eh:
                m_del(self, eh)
            self._inst_ptr = self.C_TYPE_PTR()

    def at_eigen_base_index(self, row, col=0):
        """
        Get the value at the specified index in the base Eigen matrix.

        **Note:** *The base Eigen matrix may not be the same shape as the
        current instance as this might be a sliced view of the base matrix.*

        :param row: Row of the value
        :param col: Column of the value

        :return: Value at the specified index.

        """
        assert 0 <= row < self.shape[0], "Row out of range"
        assert 0 <= col < self.shape[1], "Col out of range"
        f = self.VITAL_LIB[self._func_map['get']]
        f.argtypes = [self.C_TYPE_PTR, c_ptrdiff_t, c_ptrdiff_t,
                      VitalErrorHandle.C_TYPE_PTR]
        f.restype = self._c_type
        with VitalErrorHandle() as eh:
            return f(self, row, col, eh)
