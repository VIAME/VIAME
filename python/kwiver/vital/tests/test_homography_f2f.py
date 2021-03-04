"""
ckwg +31
Copyright 2020 by Kitware, Inc.
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

Tests for python F2FHomography interface

"""

import nose.tools as nt
import numpy as np

from kwiver.vital.types import HomographyD, HomographyF
from kwiver.vital.types.homography_f2f import *


class TestF2FHomography(object):
    def check_properties(self, f2f_hom, mat, from_id, to_id, prec=6):
        f2f_mat = f2f_hom.homography.matrix()
        np.testing.assert_array_almost_equal(f2f_mat, mat, prec)
        nt.assert_equal(f2f_hom.from_id, from_id)
        nt.assert_equal(f2f_hom.to_id, to_id)

    def test_init_from_frame(self):
        ident_hom = HomographyD()
        f1 = F2FHomography(5)
        f2 = F2FHomography(-7)
        f3 = F2FHomography(0)

        self.check_properties(f1, ident_hom.matrix(), 5, 5)
        self.check_properties(f2, ident_hom.matrix(), -7, -7)
        self.check_properties(f3, ident_hom.matrix(), 0, 0)

    def test_init_from_float_mat(self):
        mat = np.random.rand(3, 3)
        f = F2FHomography.from_floats(mat, 0, 5)
        self.check_properties(f, mat, 0, 5)

    def test_init_from_double_mat(self):
        mat = np.random.rand(3, 3)
        f = F2FHomography.from_doubles(mat, 0, 5)
        self.check_properties(f, mat, 0, 5, prec=15)

    def test_init_from_homography(self):
        hd = HomographyD()
        hf = HomographyF()

        f2f_d = F2FHomography(hd, 0, 5)
        f2f_f = F2FHomography(hf, 0, 5)

        self.check_properties(f2f_d, hd.matrix(), 0, 5, prec=15)
        self.check_properties(f2f_f, hf.matrix(), 0, 5)

        hd = HomographyD.random()
        hf = HomographyF.random()

        f2f_d = F2FHomography(hd, 5, 0)
        f2f_f = F2FHomography(hf, 5, 0)

        self.check_properties(f2f_d, hd.matrix(), 5, 0, prec=15)
        self.check_properties(f2f_f, hf.matrix(), 5, 0)

    def test_copy_construct(self):
        # Copy from default constructor
        f2f = F2FHomography(10)
        f2f_copy = F2FHomography(f2f)
        self.check_properties(f2f_copy, f2f.homography.matrix(), 10, 10, prec=15)

        # Copy from specified matrix for doubles and floats
        f2f_d = F2FHomography(HomographyD.random(), 0, 5)
        f2f_f = F2FHomography(HomographyF.random(), 0, 5)

        f2f_d_copy = F2FHomography(f2f_d)
        f2f_f_copy = F2FHomography(f2f_f)

        self.check_properties(f2f_d_copy, f2f_d.homography.matrix(), 0, 5, prec=15)
        self.check_properties(f2f_f_copy, f2f_f.homography.matrix(), 0, 5)

    def test_inverse(self):
        # Test inverse for identity matrix
        f2f = F2FHomography(10)
        self.check_properties(f2f.inverse(), np.identity(3), 10, 10, prec=15)

        h_d = HomographyD.random()
        h_f = HomographyF.random()
        f2f_d = F2FHomography(h_d, 0, 5)
        f2f_f = F2FHomography(h_f, 0, 5)
        self.check_properties(f2f_d.inverse(), h_d.inverse().matrix(), 5, 0, prec=15)
        self.check_properties(f2f_f.inverse(), h_f.inverse().matrix(), 5, 0)

    def test_mul_ident(self):
        # Identity * identity = identity
        ident = F2FHomography(10)
        self.check_properties(ident * ident, np.identity(3), 10, 10, prec=15)

        # Identity * F = F
        f2f_d = F2FHomography(HomographyD.random(), 0, 10)
        f2f_f = F2FHomography(HomographyF.random(), 0, 10)
        mat_d = f2f_d.homography.matrix()
        mat_f = f2f_f.homography.matrix()

        self.check_properties(ident * f2f_d, mat_d, 0, 10, prec=15)
        self.check_properties(ident * f2f_f, mat_f, 0, 10)

    def test_mul(self):
        # Doubles
        f2f_d1 = F2FHomography(HomographyD.random(), 0, 5)
        f2f_d2 = F2FHomography(HomographyD.random(), 10, 0)
        mat1 = f2f_d1.homography.matrix()
        mat2 = f2f_d2.homography.matrix()
        self.check_properties(f2f_d1 * f2f_d2, np.dot(mat1, mat2), 10, 5, prec=15)

        # Floats
        f2f_f1 = F2FHomography(HomographyF.random(), 0, 5)
        f2f_f2 = F2FHomography(HomographyF.random(), 10, 0)
        mat1 = f2f_f1.homography.matrix()
        mat2 = f2f_f2.homography.matrix()
        self.check_properties(f2f_f1 * f2f_f2, np.dot(mat1, mat2), 10, 5)

    def test_mul_different_types(self):
        f2f_d = F2FHomography(HomographyD.random(), 0, 5)
        f2f_f = F2FHomography(HomographyF.random(), 10, 0)
        mat_d = f2f_d.homography.matrix()
        mat_f = f2f_f.homography.matrix()
        self.check_properties(f2f_d * f2f_f, np.dot(mat_d, mat_f), 10, 5, prec=15)

    def test_mul_exception(self):
        # If LHS.from_id != RHS.to_id, __mul__ should raise an exception
        f1 = F2FHomography(HomographyD.random(), 0, 5)
        f2 = F2FHomography(HomographyD.random(), 10, 1)

        exp_err_msg = "Homography frame identifiers do not match up"

        # 0 != 1, so this should throw
        with nt.assert_raises_regexp(RuntimeError, exp_err_msg):
            f1 * f2

    def check_each_element_equal(self, f2f, mat, prec=6):
        for i in range(3):
            for j in range(3):
                nt.assert_almost_equal(f2f.get(i, j), mat[i, j], prec)

    def test_get_ident(self):
        f2f = F2FHomography(10)
        self.check_each_element_equal(f2f, np.identity(3), prec=15)

    def test_get(self):
        f2f_d = F2FHomography(HomographyD.random(), 0, 5)
        f2f_f = F2FHomography(HomographyF.random(), 10, 0)
        mat_d = f2f_d.homography.matrix()
        mat_f = f2f_f.homography.matrix()

        self.check_each_element_equal(f2f_d, mat_d, prec=15)
        self.check_each_element_equal(f2f_f, mat_f)

    def test_get_oob(self):
        f2fs = F2FHomography(5), F2FHomography(HomographyD.random(), 5, 10)
        exp_err_msg = "Tried to perform get\\(\\) out of bounds"
        for f2f in f2fs:
            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2f.get(3, 0)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2f.get(-4, 0)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2f.get(0, 3)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2f.get(0, -4)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2f.get(5, 5)

            with nt.assert_raises_regexp(IndexError, exp_err_msg):
                f2f.get(-6, -6)

    def check_str(self, f2f):
        m_out_expected = f2f.homography.matrix().flatten()

        to_str = str(f2f)
        str_split = to_str.split()

        nt.assert_equals(int(str_split[0]), f2f.from_id)
        nt.assert_equals(str_split[1], "->")
        nt.assert_equals(int(str_split[2]), f2f.to_id)

        # Now the matrix
        m_out = [float(x) for x in str_split[3:]]
        np.testing.assert_array_almost_equal(m_out, m_out_expected)

    def print_hom_and_mat(self, f):
        print("F2FHomography string:\n", str(f))
        print("Matrix values:\n", f.homography.matrix())

    def test_to_str_ident(self):
        f = F2FHomography(10)
        print("\nPrinting identity f2f_homography")
        self.print_hom_and_mat(f)
        self.check_str(f)

    def test_to_str(self):
        f2f_d = F2FHomography(HomographyD.random(), 0, 5)
        f2f_f = F2FHomography(HomographyF.random(), 10, 0)

        print("\nPrinting double f2f_homography")
        self.print_hom_and_mat(f2f_d)
        self.check_str(f2f_d)

        print("\nPrinting float f2f_homography")
        self.print_hom_and_mat(f2f_f)
        self.check_str(f2f_f)
