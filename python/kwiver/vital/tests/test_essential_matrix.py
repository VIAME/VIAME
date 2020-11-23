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

Tests for Python interface to vital::essential_matrix

"""

from kwiver.vital.types.essential_matrix import BaseEssentialMatrix, \
                                                EssentialMatrixD, EssentialMatrixF
from kwiver.vital.types import RotationD, RotationF

import nose.tools as nt
import numpy as np


class TestVitalEssentialMatrix(object):
    def setUp(self):
        # Matrices
        self.rng = np.random.default_rng()
        self.rand_float_mat = 10 * self.rng.random((3, 3), dtype="f") - 5
        self.rand_double_mat = 10 * self.rng.random((3, 3), dtype="d") - 5

        # Rotation and vector
        self.rot_d = RotationD([1.0, 2.0, 3.0])
        self.rot_f = RotationF([1.0, 2.0, 3.0])
        self.translation = np.array([-1.0, 1.0, 4.0])

    def test_no_construct_base(self):
        with nt.assert_raises_regexp(
            TypeError,
            "kwiver.vital.types.essential_matrix.BaseEssentialMatrix: No constructor defined!",
        ):
            BaseEssentialMatrix()

    def _create_ems(self):
        return (
            EssentialMatrixF(self.rot_f, self.translation),
            EssentialMatrixD(self.rot_d, self.translation),
        )

    def test_create(self):
        # float
        emf = EssentialMatrixF(self.rand_float_mat)
        emf = EssentialMatrixF(self.rot_f, self.translation)
        emf_cpy = EssentialMatrixF(emf)

        # double
        emd = EssentialMatrixD(self.rand_double_mat)
        emd = EssentialMatrixD(self.rot_d, self.translation)
        emd_cpy = EssentialMatrixD(emd)

        # Copy constructor from other type
        emd_as_float = EssentialMatrixF(emd)
        emf_as_double = EssentialMatrixD(emf)

    def test_type_name(self):
        emf, emd = self._create_ems()

        emd_as_float = EssentialMatrixF(emd)
        emf_as_double = EssentialMatrixD(emf)

        nt.assert_equals(emf.type_name, "f")
        nt.assert_equals(emd_as_float.type_name, "f")

        nt.assert_equals(emd.type_name, "d")
        nt.assert_equals(emf_as_double.type_name, "d")

    def test_clone_matrix_equal(self):
        emf, emd = self._create_ems()

        np.testing.assert_array_almost_equal(emf.matrix(), emf.clone().matrix(), 6)
        np.testing.assert_array_almost_equal(emd.matrix(), emd.clone().matrix(), 15)

        emd_as_float = EssentialMatrixF(emd)
        emf_as_double = EssentialMatrixD(emf)

        np.testing.assert_array_almost_equal(emf_as_double.matrix(), emf_as_double.clone().matrix(), 6)
        np.testing.assert_array_almost_equal(emd_as_float.matrix(), emd_as_float.clone().matrix(), 6)

    def test_clone_is_same_type(self):
        emf, emd = self._create_ems()

        nt.ok_(isinstance(emf.clone(), EssentialMatrixF))
        nt.ok_(isinstance(emd.clone(), EssentialMatrixD))

    def check_arrays_similar(self, m1, m2):
        are_equal = np.allclose(m1, m2)
        differ_by_minus = np.allclose(m1, -m2)
        return are_equal or differ_by_minus

    def properties_helper(self, em, rot, prec):
        w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t_norm = self.translation / np.linalg.norm(self.translation)

        u, s, vh = np.linalg.svd(em.matrix())
        np.testing.assert_array_almost_equal(s, [1, 1, 0])
        nt.assert_almost_equal(np.linalg.norm(em.translation()), 1, prec)

        t_extracted = u[:, 2]

        np.testing.assert_array_almost_equal(t_extracted, t_norm, prec)

        r1_extracted = np.dot(u, np.dot(w, vh))
        r2_extracted = np.dot(u, np.dot(w.T, vh))

        matches_one = self.check_arrays_similar(
            r1_extracted, rot.matrix()
        ) or self.check_arrays_similar(r2_extracted, rot.matrix())

        nt.ok_(
            matches_one,
            "Extracted rotation should match input or twisted pair for matrix of type: {}".format(
                em.type_name
            ),
        )
        print(
            "Input:",
            rot.matrix(),
            "Result (v1):",
            r1_extracted,
            "Result (v2):",
            r2_extracted,
            sep="\n",
        )

    def test_properties(self):
        emf, emd = self._create_ems()

        self.properties_helper(emf, self.rot_f, 6)
        self.properties_helper(emd, self.rot_d, 15)

    def twisted_pair_helper(self, em):
        t_norm = self.translation / np.linalg.norm(self.translation)

        r1 = em.rotation()
        r2 = em.twisted_rotation()
        t1 = em.translation()
        t2 = -t1

        rot_t_180 = RotationD(np.pi, t_norm)
        np.testing.assert_array_almost_equal(
            (rot_t_180 * r1).matrix(),
            r2.matrix(),
            6,
            err_msg="Twisted pair rotation should be 180 degree rotation around t for type {}".format(
                em.type_name
            ),
        )

        em1, em2, em3, em4 = (
            EssentialMatrixD(r1, t1),
            EssentialMatrixD(r1, t2),
            EssentialMatrixD(r2, t1),
            EssentialMatrixD(r2, t2),
        )

        m1, m2, m3, m4 = em1.matrix(), em2.matrix(), em3.matrix(), em4.matrix()
        m = em.matrix()

        nt.ok_(
            self.check_arrays_similar(m, m1),
            "Possible factorization 1 should match source for type: {}".format(
                em.type_name
            ),
        )
        nt.ok_(
            self.check_arrays_similar(m, m2),
            "Possible factorization 2 should match source for type: {}".format(
                em.type_name
            ),
        )
        nt.ok_(
            self.check_arrays_similar(m, m3),
            "Possible factorization 3 should match source for type: {}".format(
                em.type_name
            ),
        )
        nt.ok_(
            self.check_arrays_similar(m, m4),
            "Possible factorization 4 should match source for type: {}".format(
                em.type_name
            ),
        )

    def test_twisted_pair(self):
        emf, emd = self._create_ems()

        self.twisted_pair_helper(emf)
        self.twisted_pair_helper(emd)

    # The essential_matrix class inherits some public member functions
    # from its base class. These functions return
    # matrices/vectors/rotations of type double.
    # The derived classes have their own member functions
    # which return these same objects templated by their native type
    def test_typed_members(self):
        emf, emd = self._create_ems()

        # floats first

        # Check matrix
        np.testing.assert_array_almost_equal(emf.compute_matrix(), emf.matrix(), 6)

        # Check twisted rotation
        np.testing.assert_array_almost_equal(
            emf.compute_twisted_rotation().matrix(), emf.twisted_rotation().matrix(), 6
        )
        nt.ok_(isinstance(emf.compute_twisted_rotation(), RotationF))

        # Check rotation
        np.testing.assert_array_almost_equal(
            emf.get_rotation().matrix(), emf.rotation().matrix(), 6
        )
        nt.ok_(isinstance(emf.get_rotation(), RotationF))

        # Lastly check translation
        np.testing.assert_array_almost_equal(
            emf.get_translation(), emf.translation(), 6
        )

        # Now doubles
        np.testing.assert_array_almost_equal(emd.compute_matrix(), emd.matrix(), 15)

        np.testing.assert_array_almost_equal(
            emd.compute_twisted_rotation().matrix(), emd.twisted_rotation().matrix(), 15
        )
        nt.ok_(isinstance(emd.compute_twisted_rotation(), RotationD))

        np.testing.assert_array_almost_equal(
            emd.get_rotation().matrix(), emd.rotation().matrix(), 15
        )
        nt.ok_(isinstance(emd.get_rotation(), RotationD))

        np.testing.assert_array_almost_equal(
            emd.get_translation(), emd.translation(), 15
        )

    def test_to_str(self):
        emf, emd = self._create_ems()
        m_out_expected = emf.matrix().flatten()

        # Floats
        # Reads the values from the string representation back
        # to floats
        m_out = [float(x) for x in str(emf).split()]

        # Now compare the actual matrix with the matrix retrieved from the string
        np.testing.assert_almost_equal(m_out, m_out_expected, 5)
        print("\nEssentialMatrixF string:\n", str(emf))
        print("Matrix values:\n", emf.matrix())

        # Doubles
        m_out_expected = emd.matrix().flatten()

        m_out = [float(x) for x in str(emd).split()]

        np.testing.assert_almost_equal(m_out, m_out_expected, 5)
        print("\nEssentialMatrixD string:\n", str(emd))
        print("Matrix values:\n", emd.matrix())

    def check_each_element_equal(self, em, mat, prec):
        for i in range(3):
            for j in range(3):
                np.testing.assert_almost_equal(
                    em[i, j],
                    mat[i, j],
                    prec,
                    err_msg="Element inequality at row {} column {}".format(i, j),
                )

    def test_getitem(self):
        emf, emd = self._create_ems()
        trans = self.translation  / np.linalg.norm(self.translation)
        cross_d = np.array([[0,        -trans[2], trans[1]],
                            [trans[2],  0,        -trans[0]],
                            [-trans[1], trans[0], 0]], dtype='d')

        cross_f = np.array(cross_d, dtype='f')

        # Check matches expected
        self.check_each_element_equal(emf, np.dot(cross_f, emf.get_rotation().matrix()), 6)
        self.check_each_element_equal(emd, np.dot(cross_d, emd.get_rotation().matrix()), 15)

        # Check matches result of matrix()
        self.check_each_element_equal(emf, emf.compute_matrix(), 6)
        self.check_each_element_equal(emd, emd.compute_matrix(), 15)

    def test_get_item_oob(self):
        for em in self._create_ems():
            with nt.assert_raises(IndexError):
                em[-1, 0]

            with nt.assert_raises(IndexError):
                em[3, 0]

            with nt.assert_raises(IndexError):
                em[0, -1]

            with nt.assert_raises(IndexError):
                em[0, 3]

            with nt.assert_raises(IndexError):
                em[-1, 3]

            with nt.assert_raises(IndexError):
                em[-1, -1]

            with nt.assert_raises(IndexError):
                em[3, 3]
