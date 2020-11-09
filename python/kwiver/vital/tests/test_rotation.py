"""
ckwg +31
Copyright 2016-2020 by Kitware, Inc.
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

Tests for vital.types.Rotation class

"""
from __future__ import print_function
import math
import unittest

import nose.tools
import numpy

from kwiver.vital.types import rotation, RotationD, RotationF


def array_normalize(a, dtype=None):
    a = numpy.asarray(a, dtype)
    return (a / numpy.linalg.norm(a)).tolist()


class TestVitalRotation(unittest.TestCase):
    def test_new_default(self):
        # That these even construct
        rot_d = RotationD()
        nose.tools.assert_equal(rot_d.type_name, "d")

        rot_f = RotationF()
        nose.tools.assert_equal(rot_f.type_name, "f")

    def test_eq(self):
        # Identities should equal
        r1 = RotationD()
        r2 = RotationD()
        nose.tools.assert_equal(r1, r2)

        r3 = RotationD([1, 2, 3, 4])
        r4 = RotationD([1, 2, 3, 4])
        nose.tools.assert_equal(r3, r4)
        nose.tools.assert_false(r1 == r3)

        r1 = RotationF()
        r2 = RotationF()
        nose.tools.assert_equal(r1, r2)

        r3 = RotationF([1, 2, 3, 4])
        r4 = RotationF([1, 2, 3, 4])
        nose.tools.assert_equal(r3, r4)
        nose.tools.assert_false(r1 == r3)

        r1 = RotationD([1, 2, 3, 4])
        r2 = RotationD([-1, -2, -3, -4])
        assert r1.angle_from(r2) < 1e-12

    def test_not_eq(self):
        # Identities should equal
        r1 = RotationD()
        r2 = RotationD()
        nose.tools.assert_false(r1 != r2)

        r3 = RotationD([1, 2, 3, 4])
        r4 = RotationD([1, 2, 3, 4])
        nose.tools.assert_false(r3 != r4)
        nose.tools.ok_(r1 != r3)

        r1 = RotationF()
        r2 = RotationF()
        nose.tools.assert_false(r1 != r2)

        r3 = RotationF([1, 2, 3, 4])
        r4 = RotationF([1, 2, 3, 4])
        nose.tools.assert_false(r3 != r4)
        nose.tools.ok_(r1 != r3)

    def test_to_matrix(self):
        # Default value should be identity
        rot_d = RotationD()
        numpy.testing.assert_array_equal(rot_d.matrix(), numpy.eye(3))

        rot_f = RotationF()
        numpy.testing.assert_array_equal(rot_f.matrix(), numpy.eye(3))

    def test_to_quaternion(self):
        rot_d = RotationD()
        numpy.testing.assert_array_equal(rot_d.quaternion(), [0, 0, 0, 1])

        rot_f = RotationF()
        numpy.testing.assert_array_equal(rot_f.quaternion(), [0, 0, 0, 1])

    def test_to_axis_angle(self):
        # expected identity: [0,0,1] and 0
        ident_axis = [0, 0, 1]
        ident_angle = 0

        rot_d = RotationD()
        rot_f = RotationF()

        numpy.testing.assert_equal(rot_d.axis(), ident_axis)
        nose.tools.assert_equal(rot_d.angle(), ident_angle)

        numpy.testing.assert_equal(rot_f.axis(), ident_axis)
        nose.tools.assert_equal(rot_f.angle(), ident_angle)

    def test_to_rodrigues(self):
        # rodrigues identity: [0,0,0]
        ident_rod = [0, 0, 0]

        rot_d = RotationD()
        rot_f = RotationF()

        rod = rot_d.rodrigues()
        numpy.testing.assert_equal(rod, ident_rod)

        rod = rot_f.rodrigues()
        numpy.testing.assert_equal(rod, ident_rod)

    def test_to_ypr(self):
        # ypr identity: (pi/2, 0, pi)
        ident_ypr = (math.pi / 2, 0, -math.pi)

        rot_d = RotationD()
        rot_f = RotationF()

        numpy.testing.assert_almost_equal(rot_d.yaw_pitch_roll(), ident_ypr, 15)

        numpy.testing.assert_almost_equal(rot_f.yaw_pitch_roll(), ident_ypr)

    def test_from_rotation(self):
        r = RotationD()
        r_cpy = RotationD(r)
        nose.tools.ok_(r == r_cpy)

        r = RotationD([1, 2, 3, 4])
        r_cpy = RotationD(r)
        nose.tools.ok_(r == r_cpy)

        r = RotationF()
        r_cpy = RotationF(r)
        nose.tools.ok_(r == r_cpy)

        r = RotationF([1, 2, 3, 4])
        r_cpy = RotationF(r)
        nose.tools.ok_(r == r_cpy)

    def test_from_rotation_other_type(self):
        r = RotationD()
        r_cpy = RotationF(r)
        numpy.testing.assert_array_almost_equal(r.quaternion(), r_cpy.quaternion(), 6)

        r = RotationD([1, 2, 3, 4])
        r_cpy = RotationF(r)
        numpy.testing.assert_array_almost_equal(r.quaternion(), r_cpy.quaternion(), 6)

        r = RotationF()
        r_cpy = RotationD(r)
        numpy.testing.assert_array_almost_equal(r.quaternion(), r_cpy.quaternion(), 6)

        r = RotationF([1, 2, 3, 4])
        r_cpy = RotationD(r)
        numpy.testing.assert_array_almost_equal(r.quaternion(), r_cpy.quaternion(), 6)


    def test_from_quaternion(self):
        q = array_normalize([+2, -1, -3, +0], float)
        r = RotationD(q)
        numpy.testing.assert_equal(r.quaternion(), q)

    def test_from_rodrigues(self):
        rod_list_1 = [0, 0, 0]

        r1 = RotationD(rod_list_1)
        numpy.testing.assert_equal(r1.rodrigues(), rod_list_1)

        # This one will get normalized by magnitude in rotation instance
        # This vector's is less than 2*pi, so we should expect this vector to be
        #   returned as is.
        rod2 = numpy.array([2, -1, 0.5])
        nod2_normed = array_normalize(rod2)
        print("r2 2-norm:", numpy.linalg.norm(rod2))
        print("r2-normed:", nod2_normed)

        r2 = RotationD(rod2)
        numpy.testing.assert_array_almost_equal(
            r2.rodrigues(), rod2, decimal=14,  # 1e-14
        )

    def test_from_aa(self):
        # Axis should come out of rotation normalized
        angle = 0.8
        axis = [-3, 2, 1]
        axis_norm = array_normalize(axis)

        r = RotationD(angle, axis)
        nose.tools.assert_equal(angle, r.angle())
        numpy.testing.assert_equal(axis_norm, r.axis())

    def test_from_ypr(self):
        y = 1.2
        p = 0.3
        r = -1.0

        # XXX
        rot = RotationD(y, p, r)
        ry, rp, rr = rot.yaw_pitch_roll()
        nose.tools.assert_almost_equal(y, ry, 14)
        nose.tools.assert_almost_equal(p, rp, 14)
        nose.tools.assert_almost_equal(r, rr, 14)

        # 0XX
        rot = RotationD(0, p, r)
        ry, rp, rr = rot.yaw_pitch_roll()
        nose.tools.assert_almost_equal(0, ry, 14)
        nose.tools.assert_almost_equal(p, rp, 14)
        nose.tools.assert_almost_equal(r, rr, 14)

        # X0X
        rot = RotationD(y, 0, r)
        ry, rp, rr = rot.yaw_pitch_roll()
        nose.tools.assert_almost_equal(y, ry, 14)
        nose.tools.assert_almost_equal(0, rp, 14)
        nose.tools.assert_almost_equal(r, rr, 14)

        # XX0
        rot = RotationD(y, p, 0)
        ry, rp, rr = rot.yaw_pitch_roll()
        nose.tools.assert_almost_equal(y, ry, 14)
        nose.tools.assert_almost_equal(p, rp, 14)
        nose.tools.assert_almost_equal(0, rr, 14)

        # 00X
        rot = RotationD(0, 0, r)
        ry, rp, rr = rot.yaw_pitch_roll()
        nose.tools.assert_almost_equal(0, ry, 14)
        nose.tools.assert_almost_equal(0, rp, 14)
        nose.tools.assert_almost_equal(r, rr, 14)

        # 0X0
        rot = RotationD(0, p, 0)
        ry, rp, rr = rot.yaw_pitch_roll()
        nose.tools.assert_almost_equal(0, ry, 14)
        nose.tools.assert_almost_equal(p, rp, 14)
        nose.tools.assert_almost_equal(0, rr, 14)

        # X00
        rot = RotationD(y, 0, 0)
        ry, rp, rr = rot.yaw_pitch_roll()
        nose.tools.assert_almost_equal(y, ry, 14)
        nose.tools.assert_almost_equal(0, rp, 14)
        nose.tools.assert_almost_equal(0, rr, 14)

        # 000
        rot = RotationD(0, 0, 0)
        ry, rp, rr = rot.yaw_pitch_roll()
        nose.tools.assert_almost_equal(0, ry, 14)
        nose.tools.assert_almost_equal(0, rp, 14)
        nose.tools.assert_almost_equal(0, rr, 14)

    def test_from_matrix(self):
        # Create a non-identity matrix from a different constructor that we
        #   assume works
        # Create new rotation with that matrix.
        # New rotation to_matrix method should produce the same matrix
        pre_r = RotationD([+2, -1, -3, +0])
        mat = pre_r.matrix()
        r = RotationD(mat)
        numpy.testing.assert_allclose(mat, r.matrix(), 1e-15)

    def test_inverse(self):
        # quaternion calc from:
        #   https://www.wolframalpha.com/input/?i=quaternion:+0%2B2i-j-3k&lk=3
        r = RotationD([+2, -1, -3, +0])
        r_inv = r.inverse()
        e_inv = array_normalize([-1 / 7.0, +1 / 14.0, +3 / 14.0, 0])
        numpy.testing.assert_allclose(r_inv.quaternion(), e_inv, 1e-15)

        r = RotationF([+2, -1, -3, +0])
        r_inv = r.inverse()
        numpy.testing.assert_allclose(r_inv.quaternion(), e_inv, 1e-7)

    def test_mul(self):
        # Normalize quaternaion vector.
        expected_quat = array_normalize([+2.0, -1.0, -3.0, +0.0])

        r_ident_d = RotationD()
        r_ident_f = RotationF()
        r_other_d = RotationD(expected_quat)
        r_other_f = RotationF(expected_quat)

        r_res_d = r_ident_d * r_other_d
        nose.tools.assert_is_not(r_other_d, r_res_d)
        numpy.testing.assert_equal(r_res_d, r_other_d)
        numpy.testing.assert_equal(r_res_d.quaternion(), expected_quat)

        r_res_f = r_ident_f * r_other_f
        nose.tools.assert_is_not(r_other_f, r_res_f)
        numpy.testing.assert_equal(r_res_f, r_other_f)
        numpy.testing.assert_allclose(r_res_f.quaternion(), expected_quat, 1e-7)

    def test_mul_vector(self):
        vec = [1, 0, 0]
        vec_expected = [0, 1, 0]

        r_axis = [0, 0, 1]
        r_angle = math.pi / 2.0
        r = RotationD(r_angle, r_axis)

        vec_rotated = r * vec
        numpy.testing.assert_array_almost_equal(vec_expected, vec_rotated)

    def test_interpolation(self):
        x_d = RotationD(0, [1, 0, 0])
        y_d = RotationD(math.pi / 2, [0, 1, 0])
        r_d = RotationD(math.pi / 4, [0, 1, 0])

        x_f = RotationF(0, [1, 0, 0])
        y_f = RotationF(math.pi / 2, [0, 1, 0])
        r_f = RotationF(math.pi / 4, [0, 1, 0])

        z_d = rotation.interpolate_rotation(x_d, y_d, 0.5)
        z_f = rotation.interpolate_rotation(x_f, y_f, 0.5)
        nose.tools.assert_almost_equal((z_d.inverse() * r_d).angle(), 0, 14)
        nose.tools.assert_almost_equal((z_f.inverse() * r_f).angle(), 0, 6)

    def test_interpolated_rotations(self):
        x = RotationD(0, [1, 0, 0])
        a = math.pi / 2
        y = RotationD(a, [0, 1, 0])
        i_list = rotation.interpolated_rotations(x, y, 3)
        nose.tools.assert_equal([i.type_name for i in i_list], ["d"] * 3)

        i0_e_axis, i0_e_angle = [0, 1, 0], a * 0.25
        i1_e_axis, i1_e_angle = [0, 1, 0], a * 0.50
        i2_e_axis, i2_e_angle = [0, 1, 0], a * 0.75

        numpy.testing.assert_almost_equal(i_list[0].axis(), i0_e_axis, 14)
        numpy.testing.assert_almost_equal(i_list[0].angle(), i0_e_angle, 14)

        numpy.testing.assert_almost_equal(i_list[1].axis(), i1_e_axis, 14)
        numpy.testing.assert_almost_equal(i_list[1].angle(), i1_e_angle, 14)

        numpy.testing.assert_almost_equal(i_list[2].axis(), i2_e_axis, 14)
        numpy.testing.assert_almost_equal(i_list[2].angle(), i2_e_angle, 14)
