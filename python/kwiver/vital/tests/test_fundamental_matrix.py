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

Tests for Python interface to vital::fundamental_matrix

"""

from kwiver.vital.types import FundamentalMatrixF, FundamentalMatrixD

import nose.tools as nt
import numpy as np


class TestVitalFundamentalMatrix(object):
    def setUp(self):
        self.rng = np.random.default_rng()
        self.rand_float_mat = 10 * self.rng.random((3, 3), dtype="f") - 5
        self.rand_double_mat = 10 * self.rng.random((3, 3), dtype="d") - 5

    def _create_fms(self):
        fmf = FundamentalMatrixF(self.rand_float_mat)
        fmd = FundamentalMatrixD(self.rand_double_mat)
        return [
            fmf,
            FundamentalMatrixF(fmf),
            FundamentalMatrixF(fmd),
            FundamentalMatrixF(self.rand_double_mat),
            fmd,
            FundamentalMatrixD(fmd),
            FundamentalMatrixD(fmf),
            FundamentalMatrixD(self.rand_float_mat),
        ]

    def test_create(self):
        # float
        fmf = FundamentalMatrixF(self.rand_float_mat)
        fmf_cpy = FundamentalMatrixF(fmf)

        # double
        fmd = FundamentalMatrixD(self.rand_double_mat)
        fmd_cpy = FundamentalMatrixD(fmd)

        # Copy constructor from other type
        fmd_as_float = FundamentalMatrixF(fmd)
        fmf_as_double = FundamentalMatrixD(fmf)

    def test_create_from_matrix_other_type(self):
        # Create fundamental_matrix<float> from double array
        fmf = FundamentalMatrixF(self.rand_double_mat)

        # Create fundamental_matrix<double> from float array
        fmd = FundamentalMatrixD(self.rand_float_mat)

    def test_type_name(self):
        # float
        fmf = FundamentalMatrixF(self.rand_float_mat)
        fmf_cpy = FundamentalMatrixF(fmf)
        nt.assert_equals(fmf.type_name, "f")
        nt.assert_equals(fmf_cpy.type_name, "f")

        # double
        fmd = FundamentalMatrixD(self.rand_double_mat)
        fmd_cpy = FundamentalMatrixD(fmd)
        nt.assert_equals(fmd.type_name, "d")
        nt.assert_equals(fmd_cpy.type_name, "d")

        # Copy constructor from other fundamental_matrix type
        fmd_as_float = FundamentalMatrixF(fmd)
        fmf_as_double = FundamentalMatrixD(fmf)
        nt.assert_equals(fmd_as_float.type_name, "f")
        nt.assert_equals(fmf_as_double.type_name, "d")

        # fundamental_matrix<float> from double array and vice versa
        fmf = FundamentalMatrixF(self.rand_double_mat)
        fmd = FundamentalMatrixD(self.rand_float_mat)
        nt.assert_equals(fmf.type_name, "f")
        nt.assert_equals(fmd.type_name, "d")

    def test_clone_matrix_equal(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        fmf = FundamentalMatrixF(m)
        fmd = FundamentalMatrixD(m)
        np.testing.assert_array_almost_equal(fmf.matrix(), fmf.clone().matrix(), 6)
        np.testing.assert_array_almost_equal(fmd.matrix(), fmd.clone().matrix(), 15)

        fmf_cpy = FundamentalMatrixF(fmf)
        fmd_cpy = FundamentalMatrixD(fmd)
        np.testing.assert_array_almost_equal(
            fmf_cpy.matrix(), fmf_cpy.clone().matrix(), 6
        )
        np.testing.assert_array_almost_equal(
            fmd_cpy.matrix(), fmd_cpy.clone().matrix(), 15
        )

        fmd_as_float = FundamentalMatrixF(fmd)
        fmf_as_double = FundamentalMatrixD(fmf)
        np.testing.assert_array_almost_equal(
            fmd_as_float.matrix(), fmd_as_float.clone().matrix(), 6
        )
        np.testing.assert_array_almost_equal(
            fmf_as_double.matrix(), fmf_as_double.clone().matrix(), 15
        )

    def test_clone_is_same_type(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        fmf = FundamentalMatrixF(m)
        fmd = FundamentalMatrixD(m)
        nt.ok_(isinstance(fmf.clone(), FundamentalMatrixF))
        nt.ok_(isinstance(fmd.clone(), FundamentalMatrixD))

        fmf_cpy = FundamentalMatrixF(fmf)
        fmd_cpy = FundamentalMatrixD(fmd)
        nt.ok_(isinstance(fmf_cpy.clone(), FundamentalMatrixF))
        nt.ok_(isinstance(fmd_cpy.clone(), FundamentalMatrixD))

        fmd_as_float = FundamentalMatrixF(fmd)
        fmf_as_double = FundamentalMatrixD(fmf)
        nt.ok_(isinstance(fmd_as_float.clone(), FundamentalMatrixF))
        nt.ok_(isinstance(fmf_as_double.clone(), FundamentalMatrixD))

    def svd_helper(self, m):
        u, s, vh = np.linalg.svd(m)
        s[2] = 0
        s /= np.linalg.norm(s)
        s = np.diag(s)
        return np.dot(u, np.dot(s, vh))

    def test_matrix(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m_exp = self.svd_helper(m)

        fmf = FundamentalMatrixF(m)
        fmd = FundamentalMatrixD(m)
        np.testing.assert_array_almost_equal(fmf.matrix(), m_exp, 6)
        np.testing.assert_array_almost_equal(fmd.matrix(), m_exp, 15)

        fmf_cpy = FundamentalMatrixF(fmf)
        fmd_cpy = FundamentalMatrixD(fmd)
        np.testing.assert_array_almost_equal(fmf_cpy.matrix(), m_exp, 6)
        np.testing.assert_array_almost_equal(fmd_cpy.matrix(), m_exp, 15)

        # Use accuracy of 6 for both due to loss of precision from conversions
        fmd_as_float = FundamentalMatrixF(fmd)
        fmf_as_double = FundamentalMatrixD(fmf)
        np.testing.assert_array_almost_equal(fmd_as_float.matrix(), m_exp, 6)
        np.testing.assert_array_almost_equal(fmf_as_double.matrix(), m_exp, 6)

    def test_to_str(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        fmf = FundamentalMatrixF(m)
        m_out_expected = fmf.matrix().flatten()

        # Reads the values from the string representation back
        # to floats
        m_out = [float(x) for x in str(fmf).split()]

        # Now compare the actual matrix with the matrix retrieved from the string
        np.testing.assert_almost_equal(m_out, m_out_expected, 5)
        print("\nFundamentalMatrixF string:\n", str(fmf))
        print("Matrix values:\n", fmf.matrix())

        # Now doubles
        fmd = FundamentalMatrixD(m)
        m_out_expected = fmd.matrix().flatten()

        m_out = [float(x) for x in str(fmd).split()]

        np.testing.assert_almost_equal(m_out, m_out_expected, 6)
        print("\nFundamentalMatrixD string:\n", str(fmd))
        print("Matrix values:\n", fmd.matrix())

    def check_each_element_equal(self, fm, mat, prec):
        for i in range(3):
            for j in range(3):
                np.testing.assert_almost_equal(
                    fm[i, j],
                    mat[i, j],
                    prec,
                    err_msg="Element inequality at row {} column {}".format(i, j)
                )

    def test_getitem(self):
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m_exp = self.svd_helper(m)

        # Floats
        self.check_each_element_equal(FundamentalMatrixF(m), m_exp, 6)
        # Doubles
        self.check_each_element_equal(FundamentalMatrixD(m), m_exp, 15)

    def test_getitem_oob(self):
        for fm in self._create_fms():
            with nt.assert_raises(IndexError):
                fm[-1, 0]

            with nt.assert_raises(IndexError):
                fm[3, 0]

            with nt.assert_raises(IndexError):
                fm[0, -1]

            with nt.assert_raises(IndexError):
                fm[0, 3]

            with nt.assert_raises(IndexError):
                fm[-1, 3]

            with nt.assert_raises(IndexError):
                fm[-1, -1]

            with nt.assert_raises(IndexError):
                fm[3, 3]
