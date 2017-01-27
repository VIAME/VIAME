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

vital::similarity_<T> interface

"""
import ctypes

from vital.types import EigenArray, Rotation
from vital.util import VitalObject, OpaqueTypeCache


class Similarity (VitalObject):

    DFLT_CTYPE = ctypes.c_double

    TYPE_CACHE = OpaqueTypeCache("Similarity_")
    C_TYPE = TYPE_CACHE.new_type_getter()
    C_TYPE_PTR = TYPE_CACHE.new_ptr_getter()

    @classmethod
    def c_type(cls, ctype=DFLT_CTYPE):
        """ Get the C opaque type """
        # noinspection PyProtectedMember
        return cls.C_TYPE[ctype._type_]

    @classmethod
    def c_ptr_type(cls, ctype=DFLT_CTYPE):
        """ Get the C opaque pointer type """
        # noinspection PyProtectedMember
        return cls.C_TYPE_PTR[ctype._type_]

    @classmethod
    def from_matrix(cls, m, ctype=DFLT_CTYPE):
        """
        Create a similarity transform instance from a 4x4 matrix

        :param m: 4x4 transformation matrix
        :type m: EigenArray | collections.Iterable[float]

        :return: new Similarity instance
        :rtype: Similarity

        """
        m = EigenArray.from_iterable(m, ctype, target_shape=(4, 4))
        # noinspection PyProtectedMember
        tchar = ctype._type_
        cptr = cls._call_cfunc(
            'vital_similarity_%s_from_matrix4x4' % tchar,
            [EigenArray.c_ptr_type(4, 4, ctype)], [m],
            Similarity.c_ptr_type(ctype)
        )
        return Similarity(ctype=ctype, from_cptr=cptr)

    def __init__(self, scale=1, rotation=None, translation=None,
                 ctype=DFLT_CTYPE, from_cptr=None):
        """
        Initialize a new Similarity instance

        When constructing from an existing C-pointer, the correct ``ctype`` must
        still be provided as there is currently no way to introspect this.

        :param scale: scale factor. Default is 1.0.
        :type scale: float

        :param rotation: rotation instance. Default is identity.
        :type rotation: vital.types.Rotation

        :param translation: 3D translation vector (3x1). Default is (0, 0, 0).
        :type translation: collections.Iterable[float] | vital.types.EigenArray

        :param ctype: Underlying C data type to use. Double is used by default.
        :type ctype: ctypes._SimpleCData

        :param from_cptr: Existing C opaque instance pointer to use, preventing
            new instance construction. This should of course be a valid pointer
            to an instance. The ``ctype`` parameter must still be provided if
            the default is not accurate.
        :type from_cptr: ctypes._Pointer

        """
        self._ctype = ctype
        # noinspection PyProtectedMember
        self._tchar = ctype._type_

        # Setting concrete opaque/pointer types
        self.C_TYPE = self.__class__.C_TYPE[self._tchar]
        self.C_TYPE_PTR = self.__class__.C_TYPE_PTR[self._tchar]

        super(Similarity, self).__init__(from_cptr, scale, rotation,
                                         translation)

    def __mul__(self, other):
        if isinstance(other, Similarity):
            return self.compose(other)
        else:
            return self.transform_vector(other)

    def __eq__(self, other):
        if isinstance(other, Similarity):
            return self._call_cfunc(
                'vital_similarity_%s_are_equal' % self._tchar,
                [self.C_TYPE_PTR, self.C_TYPE_PTR],
                [self, other],
                ctypes.c_bool
            )
        return False

    def __ne__(self, other):
        return not (self == other)

    def _new(self, s, r, t):
        """
        :type s: float
        :type r: Rotation | None
        :type t: collections.Iterable[float] | None
        """
        # noinspection PyProtectedMember
        if r is None:
            r = Rotation(self._ctype)
        elif r._ctype != self._ctype:
            # Create new Rotation sharing our type
            r = Rotation.from_quaternion(r.quaternion(), self._ctype)

        if t is None:
            t = EigenArray.from_iterable((0, 0, 0), self._ctype)
        else:
            t = EigenArray.from_iterable(t, self._ctype, (3, 1))

        return self._call_cfunc(
            'vital_similarity_%s_new' % self._tchar,
            [self._ctype, Rotation.c_ptr_type(self._ctype),
             EigenArray.c_ptr_type(3, 1, self._ctype)],
            [s, r, t],
            self.C_TYPE_PTR
        )

    def _destroy(self):
        self._call_cfunc(
            'vital_similarity_%s_destroy' % self._tchar,
            [self.C_TYPE_PTR], [self]
        )

    @property
    def scale(self):
        """
        Get the scale of this similarity transformation
        :rtype: float
        """
        return self._call_cfunc(
            'vital_similarity_%s_scale' % self._tchar,
            [self.C_TYPE_PTR], [self],
            self._ctype
        )

    @property
    def rotation(self):
        """
        Get the rotation of this similarity transformation
        :rtype: Rotation
        """
        rot_ptr = self._call_cfunc(
            'vital_similarity_%s_rotation' % self._tchar,
            [self.C_TYPE_PTR], [self],
            Rotation.c_ptr_type(self._ctype)
        )
        return Rotation(self._ctype, rot_ptr)

    @property
    def translation(self):
        """
        Get the translation of this similarity transformation
        :rtype: EigenArray
        """
        trans_ptr = self._call_cfunc(
            'vital_similarity_%s_translation' % self._tchar,
            [self.C_TYPE_PTR], [self],
            EigenArray.c_ptr_type(3, 1, self._ctype)
        )
        return EigenArray(3, 1, dtype=self._ctype, from_cptr=trans_ptr)

    def inverse(self):
        """
        :return: Get the inverse of this similarity transformation
        :rtype: Similarity
        """
        sim_ptr = self._call_cfunc(
            'vital_similarity_%s_inverse' % self._tchar,
            [self.C_TYPE_PTR], [self],
            self.C_TYPE_PTR
        )
        return Similarity(ctype=self._ctype, from_cptr=sim_ptr)

    def compose(self, other):
        """
        Compose two similarities

        If the ``other`` similarity is of a different base data storage type,
        the returned similarity instance shares the data type with this instance
        over the ``other``.

        Regardless of returned data type, the similarity produced as the result
        of this composition is only as accurate and the least expressive data
        type among similarities composed. For example, if double and float
        similarities are composed and the returned similarity is of data type
        double, it will still be abject to 1e-7 floating point error due to one
        of its parents being of float type.

        :param other: Another similarity to compose with
        :type other: Similarity

        :return: New similarity instance
        :rtype: Similarity

        """
        if not isinstance(other, Similarity):
            raise ValueError("Can only compose with other similarity instances")

        # If other similarity is not of the same time, create an equivalent
        # similarity of the correct type
        if other._ctype != self._ctype:
            other = Similarity(other.scale, other.rotation, other.translation,
                               self._ctype)

        sim_ptr = self._call_cfunc(
            'vital_similarity_%s_compose' % self._tchar,
            [self.C_TYPE_PTR, self.C_TYPE_PTR],
            [self, other],
            self.C_TYPE_PTR
        )
        return Similarity(ctype=self._ctype, from_cptr=sim_ptr)

    def transform_vector(self, vec):
        """
        Transform a 3D vector with this similarity transformation

        :param vec: 3D Vector to transform
        :type vec: collections.Iterable[float]

        :return: Transformed 3D vector
        :rtype: EigenArray

        """
        vec = EigenArray.from_iterable(vec, self._ctype, (3, 1))
        t_ptr = self._call_cfunc(
            'vital_similarity_%s_vector_transform' % self._tchar,
            [self.C_TYPE_PTR, EigenArray.c_ptr_type(3, 1, self._ctype)],
            [self, vec],
            EigenArray.c_ptr_type(3, 1, self._ctype)
        )
        return EigenArray(3, 1, dtype=self._ctype, from_cptr=t_ptr)

    def as_matrix(self):
        """
        :return: similarity transformation as a 4x4 matrix
        :rtype: EigenArray
        """
        cptr = self._call_cfunc(
            'vital_similarity_%s_to_matrix4x4' % self._tchar,
            [self.C_TYPE_PTR], [self],
            EigenArray.c_ptr_type(4, 4, self._ctype)
        )
        return EigenArray(4, 4, dtype=self._ctype, from_cptr=cptr)
