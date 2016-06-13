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

Interface to the vital::rotation_ class

"""
import collections
import ctypes

import numpy

from vital.types.eigen import EigenArray
from vital.util import (
    free_void_ptr,
    OpaqueTypeCache,
    VitalErrorHandle,
    VitalObject,
)


class Rotation (VitalObject):

    # Override C opaque pointer type to ones that are dependent on size and type
    TYPE_CACHE = OpaqueTypeCache("Covariance_")
    C_TYPE = TYPE_CACHE.new_type_getter()
    C_TYPE_PTR = TYPE_CACHE.new_ptr_getter()

    # variable component in C type/function symbols
    TYPE_SPEC = "{type:s}"

    @classmethod
    def c_type(cls, ctype):
        """ Get the C opaque type """
        return cls.C_TYPE[cls._gen_spec(ctype)]

    @classmethod
    def c_ptr_type(cls, ctype=ctypes.c_double):
        """ Get the C opaque pointer type """
        return cls.C_TYPE_PTR[cls._gen_spec(ctype)]

    @classmethod
    def _gen_spec(cls, ctype):
        """ get function type specific name component """
        # noinspection PyProtectedMember
        return cls.TYPE_SPEC.format(type=ctype._type_)

    @classmethod
    def _get_c_function(cls, s, suffix):
        """ get C library function pointer for type spec and method suffix"""
        return cls.VITAL_LIB['vital_rotation_{}_{}'.format(s, suffix)]

    @classmethod
    def from_quaternion(cls, q_vec, ctype=ctypes.c_double):
        """
        Create rotation based on the given 4x1 (column-vector) quaternion
        representation whose format, `[x, y, z, w]`, represents the `w+xi+yj+zk`
        formula (see Eigen's `Quaternion` class).

        Input data is copied.

        :param q_vec: Quaternion column-vector array-like to initialize to.
        :type q_vec: collections.Iterable

        :param ctype: C data type to store rotation data in.
        :type ctype: _ctypes._SimpleCData

        :return: New rotation instance with the initialized rotation.
        :rtype: vital.types.Rotation

        :raises ValueError: The input array-like data did not conform to the
            specified target shape.

        """
        q_vec = EigenArray.from_iterable(q_vec, ctype, (4, 1))
        s = cls._gen_spec(ctype)
        r_from_q = cls._get_c_function(s, 'new_from_quaternion')
        r_from_q.argtypes = [EigenArray.c_ptr_type(4, 1, ctype),
                             VitalErrorHandle.C_TYPE_PTR]
        r_from_q.restype = cls.C_TYPE_PTR[cls.TYPE_SPEC.format(type=s)]
        with VitalErrorHandle() as eh:
            r_ptr = r_from_q(q_vec, eh)
        return Rotation(ctype, r_ptr)

    @classmethod
    def from_rodrigues(cls, r_vec, ctype=ctypes.c_double):
        """
        Create rotation based on the given 3x1 (column-vector) rodrigues
        representation.

        :param r_vec: Rodrigues 3x1 column-vector
        :type r_vec: collections.Iterable

        :param ctype: C data type to store rotation data in.
        :type ctype: _ctypes._SimpleCData

        :return: New rotation instance with the initialized rotation
        :rtype: vital.types.Rotation

        :raises ValueError: The input array-like data did not conform to the
            specified target shape.

        """
        r_vec = EigenArray.from_iterable(r_vec, ctype, (3, 1))
        s = cls._gen_spec(ctype)
        r_from_rod = cls._get_c_function(s, 'new_from_rodrigues')
        r_from_rod.argtypes = [EigenArray.c_ptr_type(3, 1, ctype),
                               VitalErrorHandle.C_TYPE_PTR]
        r_from_rod.restype = cls.C_TYPE_PTR[s]
        with VitalErrorHandle() as eh:
            r_ptr = r_from_rod(r_vec, eh)
        return Rotation(ctype, r_ptr)

    @classmethod
    def from_axis_angle(cls, axis, angle, ctype=ctypes.c_double):
        """
        Create rotation based on the given angle and axis vector (3x1
        column-vector). The axis vector will be normalized .

        :param axis: Axis column vector (3x1)
        :type axis: collections.Iterable

        :param angle: Angle of rotation about axis
        :type angle: float

        :param ctype: C data type to store rotation data in.
        :type ctype: _ctypes._SimpleCData

        :return: New rotation instance with the initialized rotation.
        :rtype: vital.types.Rotation

        :raises ValueError: The input array-like data did not conform to the
            specified target shape.

        """
        axis = EigenArray.from_iterable(axis, ctype, (3, 1))
        s = cls._gen_spec(ctype)
        r_from_aa = cls._get_c_function(s, 'new_from_axis_angle')
        r_from_aa.argtypes = [ctype,
                              EigenArray.c_ptr_type(3, 1, ctype),
                              VitalErrorHandle.C_TYPE_PTR]
        r_from_aa.restype = Rotation.C_TYPE_PTR[s]
        with VitalErrorHandle() as eh:
            r_ptr = r_from_aa(angle, axis, eh)
        return Rotation(ctype, r_ptr)

    @classmethod
    def from_ypr(cls, yaw, pitch, roll, ctype=ctypes.c_double):
        """
        Create rotation based on the given yaw, pitch and roll values.

        :param yaw: yaw value
        :type yaw: float

        :param pitch: pitch value
        :type pitch: float

        :param roll: roll value
        :type roll: float

        :param ctype: C data type to store rotation data in.
        :type ctype: _ctypes._SimpleCData

        :return: New rotation instance with the initialized rotation.
        :rtype: vital.types.Rotation

        """
        s = cls._gen_spec(ctype)
        r_from_ypr = cls._get_c_function(s, 'new_from_ypr')
        r_from_ypr.argtypes = [ctype, ctype, ctype, VitalErrorHandle.C_TYPE_PTR]
        r_from_ypr.restype = Rotation.C_TYPE_PTR[s]
        with VitalErrorHandle() as eh:
            r_ptr = r_from_ypr(yaw, pitch, roll, eh)
        return Rotation(ctype, r_ptr)

    @classmethod
    def from_matrix(cls, mat, ctype=ctypes.c_double):
        """
        Create rotation based on the given 3x3 rotation matrix.

        :param mat: Input rotation matrix.
        :type mat: collections.Iterable

        :param ctype: C data type to store rotation data in.
        :type ctype: _ctypes._SimpleCData

        :return: New rotation instance with the initialized rotation
        :rtype: vital.types.Rotation

        :raises ValueError: The input array-like data did not conform to the
            specified target shape.

        """
        mat = EigenArray.from_iterable(mat, ctype, (3, 3))
        s = cls._gen_spec(ctype)
        r_from_mat = cls._get_c_function(s, 'new_from_matrix')
        r_from_mat.argtypes = [EigenArray.c_ptr_type(3, 3, ctype),
                               VitalErrorHandle.C_TYPE_PTR]
        r_from_mat.restype = cls.C_TYPE_PTR[s]
        with VitalErrorHandle() as eh:
            r_ptr = r_from_mat(mat, eh)
        return Rotation(ctype, r_ptr)

    @classmethod
    def interpolate(cls, a, b, f):
        """
        Generate an interpolated rotation between `a` and `b` by a given
        fraction `f`.

        If rotations `a` and `b` are of different types, `b` is converted to `a`
        rotation that is the same type as `a`, and the returned rotation is of
        the same data-type as `a`.

        :param a: Rotation we are interpolating from.
        :type a: vital.types.Rotation

        :param b: Rotation we are interpolating towards.
        :type b: vital.types.Rotation

        :param f: Fractional value describing the interpolation point between
            `a` and `b`. This should be in the range (0, 1) (inclusive).
        :type f: float

        :return: New rotation that is the interpolation between `a` and `b` by
            fraction `f`.
        :type: vital.types.Rotation

        :raises ValueError: `a` or `b` are not rotation instances
        :raises ValueError: `f` is not in the range [0,1] (inclusive)

        """
        if not (isinstance(a, Rotation) and isinstance(b, Rotation)):
            raise ValueError("a and b are not rotations (given: (%s, %s))"
                             % (type(a), type(b)))
        if not 0 <= f <= 1:
            raise ValueError("f not in inclusive range [0, 1]. (given: %f)" % f)

        # if a._ctype != b._ctype, convert b into a new rotation with the same
        # type as a
        if a._ctype != b._ctype:
            cls.logger().debug("Converting `b` from type '%s' to '%s'",
                               b._ctype, a._ctype)
            b = Rotation.from_quaternion(b.quaternion(), a._ctype)

        r_interp = cls._get_c_function(cls._gen_spec(a._ctype), "interpolate")
        r_interp.argtypes = [a.C_TYPE_PTR, b.C_TYPE_PTR, a._ctype,
                             VitalErrorHandle.C_TYPE_PTR]
        r_interp.restype = a.C_TYPE_PTR
        with VitalErrorHandle() as eh:
            r_ptr = r_interp(a, b, f, eh)
        return Rotation(a._ctype, r_ptr)

    @classmethod
    def interpolated_rotations(cls, a, b, n):
        """
        Generate `n` evenly interpolated rotations in between `a` and `b`.

        We interpret `n` as an integer, e.g. if a float is given, we cast it to
        an integer, effectively flooring it.

        :param a: Rotation we are interpolating from.
        :param b: Rotation we are interpolating towards.
        :param n: Number of even interpolations in between `a` and `b` to
            generate.

        :return: Sequence of rotations between
        :rtype: list[Rotation]

        :raises ValueError: `a` or `b` are not rotation instances
        :raises ValueError: If `n` is less than 1.

        """
        if not (isinstance(a, Rotation) and isinstance(b, Rotation)):
            raise ValueError("a and b are not rotations (given: (%s, %s))"
                             % (type(a), type(b)))
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1 (given: ")

        # if a._ctype != b._ctype, convert b into a new rotation with the same
        # type as a
        if a._ctype != b._ctype:
            cls.logger().debug("Converting `b` from type '%s' to '%s'",
                               b._ctype, a._ctype)
            b = Rotation.from_quaternion(b.quaternion(), a._ctype)

        r_interp = cls._get_c_function(cls._gen_spec(a._ctype),
                                       "interpolated_rotations")
        r_interp.argtypes = [a.C_TYPE_PTR, b.C_TYPE_PTR, ctypes.c_size_t,
                             VitalErrorHandle.C_TYPE_PTR]
        r_interp.restype = ctypes.POINTER(a.C_TYPE_PTR)
        with VitalErrorHandle() as eh:
            r_arr_ptr = r_interp(a, b, n, eh)
        r_list = []
        for i in xrange(n):
            # Have to create a new pointer instance and copy the pointer value
            # at r_arr_ptr[i] into it.
            # the pointer content at index `i` in the ptr array
            # r_list.append(Rotation(a._ctype, r_arr_ptr[i]))
            rptr = a.C_TYPE_PTR(r_arr_ptr[i].contents)
            assert ctypes.addressof(rptr.contents) == ctypes.addressof(r_arr_ptr[i].contents)
            r_list.append(Rotation(a._ctype, rptr))

        # This causes the rotations extracted to be freed, too...
        free_void_ptr(r_arr_ptr)

        return r_list

    def __init__(self, c_type=ctypes.c_double, from_cptr=None):
        # Initialize C type and pointer + function map
        # noinspection PyProtectedMember
        self._spec = self._gen_spec(c_type)

        # Set concrete shape/type specific opaque pointer for this instance
        self.C_TYPE = self.__class__.C_TYPE[self._spec]
        self.C_TYPE_PTR = self.__class__.C_TYPE_PTR[self._spec]
        self._ctype = c_type

        super(Rotation, self).__init__(from_cptr)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.quaternion().flatten())

    def __eq__(self, other):
        if isinstance(other, Rotation):
            if self._ctype != other._ctype:
                # raise ValueError("Cannot test equality of two rotations of "
                #                  "different data types (%s != %s)"
                #                  % (self._ctype, other._ctype))
                self._log.debug("Converting `other` from %s into %s.",
                                other._ctype, self._ctype)
                other = Rotation.from_quaternion(other.quaternion(),
                                                 self._ctype)

            r_eq = self._get_c_function(self._spec, "are_equal")
            r_eq.argtypes = [self.C_TYPE_PTR, other.C_TYPE_PTR,
                             VitalErrorHandle.C_TYPE_PTR]
            r_eq.restype = ctypes.c_bool
            with VitalErrorHandle() as eh:
                return r_eq(self, other, eh)
        return False

    def __ne__(self, other):
        return not (self == other)

    def __mul__(self, other):
        """
        Apply this rotation to another rotation or a vector of a valid shape
        :param other:
        :return:

        :raises ValueError: The input was not a rotation of a congruent data type array-like data did not conform the
            expected 3x1 shape (column vector).

        """
        if isinstance(other, Rotation):
            return self.compose(other)
        elif isinstance(other, collections.Iterable):
            return self.rotate_vector(other)

        raise ValueError("Cannot multiply a Rotation against a '%s' type"
                         % type(other))

    def _new(self):
        # default new
        r_new = self._get_c_function(self._spec, 'new_default')
        r_new.argtypes = [VitalErrorHandle.C_TYPE_PTR]
        r_new.restype = self.C_TYPE_PTR
        with VitalErrorHandle() as eh:
            return r_new(eh)

    def _destroy(self):
        """ destroy our C pointer """
        r_del = self._get_c_function(self._spec, 'destroy')
        r_del.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        with VitalErrorHandle() as eh:
            r_del(self, eh)

    def matrix(self):
        """
        :return: this rotation as a new 3x3 matrix.
        :rtype: vital.types.EigenArray
        """
        r_to_mat = self._get_c_function(self._spec, "to_matrix")
        r_to_mat.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        r_to_mat.restype = EigenArray.c_ptr_type(3, 3, self._ctype)
        with VitalErrorHandle() as eh:
            mat_ptr = r_to_mat(self, eh)
            return EigenArray(3, 3, dtype=numpy.dtype(self._ctype),
                              from_cptr=mat_ptr, owns_data=True)

    def quaternion(self):
        """
        :return: this rotation as a new quaternion (4x1 matrix).
        :rtype: vital.types.EigenArray
        """
        r_to_q = self._get_c_function(self._spec, 'quaternion')
        r_to_q.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        r_to_q.restype = EigenArray.c_ptr_type(4, 1, self._ctype)
        with VitalErrorHandle() as eh:
            mat_ptr = r_to_q(self, eh)
            return EigenArray(4, dtype=numpy.dtype(self._ctype),
                              from_cptr=mat_ptr, owns_data=True)

    def axis(self):
        """
        :return: This rotation's axis and angle.
        :rtype: (vital.types.EigenArray, float)
        """
        r2axis = self._get_c_function(self._spec, 'axis')
        r2axis.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        r2axis.restype = EigenArray.c_ptr_type(3, 1, self._ctype)
        with VitalErrorHandle() as eh:
            mat_ptr = r2axis(self, eh)
        return EigenArray(3, dtype=numpy.dtype(self._ctype),
                          from_cptr=mat_ptr, owns_data=True)

    def angle(self):
        r2angle = self._get_c_function(self._spec, 'angle')
        r2angle.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        r2angle.restype = self._ctype
        with VitalErrorHandle() as eh:
            return r2angle(self, eh)

    def rodrigues(self):
        """
        :return: This rotation as a Rodrigues vector
        :rtype: vital.types.EigenArray
        """
        r2rod = self._get_c_function(self._spec, "rodrigues")
        r2rod.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        r2rod.restype = EigenArray.c_ptr_type(3, 1, self._ctype)
        with VitalErrorHandle() as eh:
            rod_ptr = r2rod(self, eh)
            return EigenArray(3, dtype=numpy.dtype(self._ctype),
                              from_cptr=rod_ptr, owns_data=True)

    def yaw_pitch_roll(self):
        r2ypr = self._get_c_function(self._spec, "ypr")
        r2ypr.argtypes = [self.C_TYPE_PTR,
                          ctypes.POINTER(self._ctype),  # yaw
                          ctypes.POINTER(self._ctype),  # pitch
                          ctypes.POINTER(self._ctype),  # roll
                          VitalErrorHandle.C_TYPE_PTR]
        # void return
        yaw = self._ctype()
        pitch = self._ctype()
        roll = self._ctype()
        with VitalErrorHandle() as eh:
            r2ypr(self, ctypes.byref(yaw), ctypes.byref(pitch), ctypes.byref(roll), eh)
        return yaw.value, pitch.value, roll.value

    def inverse(self):
        """
        :return: The inverse of this rotation as a new rotation instance.
        """
        r_inv = self._get_c_function(self._spec, 'inverse')
        r_inv.argtypes = [self.C_TYPE_PTR, VitalErrorHandle.C_TYPE_PTR]
        r_inv.restype = self.C_TYPE_PTR
        with VitalErrorHandle() as eh:
            r_ptr = r_inv(self, eh)
        return Rotation(self._ctype, from_cptr=r_ptr)

    def compose(self, other_rot):
        """
        Compose this rotation with another (multiply).

        This rotation is considered the left-hand operand and the given rotation
        is considered the right-hand operand.

        Result rotation will have the same data type as this rotation.

        :param other_rot: Right-hand side Rotation instance
        :type other_rot: vital.types.Rotation

        :return: new rotation that is the composition of this rotation and the
            other
        :rtype: vital.types.Rotation

        :raises ValueError: Other is not a rotation.

        """
        if not isinstance(other_rot, Rotation):
            raise ValueError("Other operand must be another rotation instance"
                             "(given: %s)" % type(other_rot))
        if self._ctype != other_rot._ctype:
            # raise ValueError("Cannot compose (multiply) two rotations of "
            #                  "different data types (%s != %s)"
            #                  % (self._ctype, other_rot._ctype))
            # Create new rotation from the given, but with our data type
            self._log.debug("Converting input rotation of type %s into "
                            "compatible type %s",
                            other_rot._ctype, self._ctype)
            other_rot = Rotation.from_quaternion(other_rot.quaternion(), self._ctype)

        r_compose = self._get_c_function(self._spec, "compose")
        r_compose.argtypes = [self.C_TYPE_PTR, other_rot.C_TYPE_PTR,
                              VitalErrorHandle.C_TYPE_PTR]
        r_compose.restype = self.C_TYPE_PTR
        with VitalErrorHandle() as eh:
            r_ptr = r_compose(self, other_rot, eh)
        return Rotation(self._ctype, r_ptr)

    def rotate_vector(self, vec):
        """
        Rotate a given 3x1 vector about this rotation, returning a new 3x1
        vector.

        Returned vector will have the same data type as this rotation.

        :param vec: 3x1 array-like to rotate
        :type vec: collections.Iterable

        :return: New 3x1 rotated vector
        :rtype: vital.types.EigenArray

        :raises ValueError: The input array-like data did not conform the
            expected 3x1 shape (column vector).

        """
        vec = EigenArray.from_iterable(vec, self._ctype, (3, 1))

        # make EigenArray out of input array if its not already
        r_rv = self._get_c_function(self._spec, "rotate_vector")
        r_rv.argtypes = [self.C_TYPE_PTR, vec.C_TYPE_PTR,
                         VitalErrorHandle.C_TYPE_PTR]
        r_rv.restype = EigenArray.c_ptr_type(3, 1, self._ctype)
        with VitalErrorHandle() as eh:
            m_ptr = r_rv(self, vec, eh)
        return EigenArray(3, dtype=numpy.dtype(self._ctype), from_cptr=m_ptr,
                          owns_data=True)
