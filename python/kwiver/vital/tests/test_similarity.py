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

tests for Similarity class

"""
from __future__ import print_function
import unittest

import nose.tools
import numpy

from kwiver.vital.types import (
    Rotation,
    Similarity
)


class TestSimiliarity (unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s = 2.4
        cls.r = Rotation.from_rodrigues([0.1, -1.5, 2.0])
        cls.r_f = Rotation.from_rodrigues([0.1, -1.5, 2.0], 'f')
        cls.t = [1, -2, 5]

    def test_new_default(self):
        s = Similarity()
        nose.tools.assert_equal(s.scale, 1)
        numpy.testing.assert_array_almost_equal(s.rotation.matrix(), Rotation().matrix())
        numpy.testing.assert_array_equal(s.translation,
                                         [0, 0, 0])

    def test_new(self):
        sim = Similarity(self.s, self.r, self.t)

        nose.tools.assert_equal(sim.scale, self.s)
        numpy.testing.assert_array_almost_equal(sim.rotation.matrix(), self.r.matrix())
        numpy.testing.assert_array_equal(sim.translation, self.t)

    #def test_new_mixed_types(self):
    #    # r and t are in double format, so try to use them to construct float
    #    # similarity inst
    #    sim = Similarity(self.s, self.r_f, self.t_f, 'f')

    #    nose.tools.assert_almost_equal(sim.scale, self.s, 6)
    #    nose.tools.assert_equal(sim.rotation, self.r)
    #    numpy.testing.assert_array_equal(sim.translation, self.t)

    def test_equals(self):
        s1 = Similarity()
        s2 = Similarity()
        nose.tools.assert_equal(s1, s2)

        s1 = Similarity(self.s, self.r, self.t)
        s2 = Similarity(self.s, self.r, self.t)
        nose.tools.assert_equal(s1, s2)

    def test_notequal(self):
        s1 = Similarity()
        s2 = Similarity(self.s, self.r, self.t)
        nose.tools.assert_not_equal(s1, s2)

    def test_get_scale(self):
        s = Similarity()
        nose.tools.assert_equal(s.scale, 1.0)

        s = Similarity(self.s, self.r, self.t)
        nose.tools.assert_equal(s.scale, self.s)

    def test_get_rotation(self):
        s = Similarity()
        numpy.testing.assert_array_almost_equal(s.rotation.matrix(), Rotation().matrix())

        s = Similarity(self.s, self.r, self.t)
        numpy.testing.assert_array_almost_equal(s.rotation.matrix(), self.r.matrix())

    def test_get_translation(self):
        s = Similarity()
        numpy.testing.assert_equal(s.translation, [0,0,0])

        s = Similarity(self.s, self.r, self.t)
        numpy.testing.assert_equal(s.translation, self.t)

    def test_convert_matrix(self):
        sim = Similarity()
        numpy.testing.assert_array_equal(sim.as_matrix(), numpy.eye(4))

        sim1 = Similarity(self.s, self.r, self.t)
        mat1 = sim1.as_matrix()
        sim2 = Similarity.from_matrix(mat1)
        mat2 = sim2.as_matrix()

        print("Sim1:", sim1.as_matrix())
        print("Sim2:", sim2.as_matrix())

        numpy.testing.assert_almost_equal(mat1, mat2, decimal=14)

    def test_compose(self):
        s1 = Similarity(self.s, self.r, self.t)
        s2 = Similarity(0.75,
                        Rotation.from_rodrigues([-0.5, -0.5, 1.0]),
                        [4, 6.5, 8])

        sim_comp = s1.compose(s2).as_matrix()
        mat_comp = numpy.dot(s1.as_matrix(), s2.as_matrix())
        print('sim12 comp:\n', sim_comp)
        print('mat comp:\n', mat_comp)
        print('sim - mat:\n', sim_comp - mat_comp)
        nose.tools.assert_almost_equal(
            numpy.linalg.norm(sim_comp - mat_comp, 2),
            0., 12
        )

    def test_compose_fail(self):
        s = Similarity(self.s, self.r, self.t)
        nose.tools.assert_raises(
            TypeError,
            s.compose,
            0
        )
        nose.tools.assert_raises(
            TypeError,
            s.compose,
            'foo'
        )
        nose.tools.assert_raises(
            TypeError,
            s.compose,
            [1, 2, 3]
        )

    def test_inverse(self):
        # Inverse of identity is itself
        s = Similarity()
        nose.tools.assert_equal(s, s.inverse())

        s = Similarity(self.s, self.r, self.t)
        s_i = s.inverse()
        i = s * s_i
        # Similarity composed with inverse should be identity
        nose.tools.assert_almost_equal(i.scale, 1., 14)
        nose.tools.assert_almost_equal(i.rotation.angle(), 0., 14)
        nose.tools.assert_almost_equal(numpy.linalg.norm(i.translation, 2),
                                       0., 12)

    def test_transform_vector(self):
        s = Similarity(self.s, self.r, self.t)

        v1 = [4, 2.1, 9.125]
        v2 = s.transform_vector(v1)
        v3 = s.inverse().transform_vector(v2)
        nose.tools.assert_false(numpy.allclose(v1, v2))
        nose.tools.assert_true(numpy.allclose(v1, v3))

        # This should also work with mult syntax
        v4 = s * v1
        v5 = s.inverse() * v4
        nose.tools.assert_false(numpy.allclose(v1, v4))
        nose.tools.assert_true(numpy.allclose(v1, v5))
