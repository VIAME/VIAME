"""
ckwg +31
Copyright 2016-2017 by Kitware, Inc.
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

Test Python interface to Eigen::Matrix wrapping via Numpy ndarray sub-class

"""

import unittest

import nose.tools as ntools
import numpy
from six.moves import range

from kwiver.vital.exceptions.eigen import VitalInvalidStaticEigenShape

from kwiver.vital.types import EigenArray


class TestVitalEigenMatrix (unittest.TestCase):

    def test_valid_static_size_init(self):
        """
        Test construction of some of the static sizes.
        """
        a = EigenArray().get_matrix() # default shape
        ntools.assert_equal(a.shape, (2, 1))

        a = EigenArray(2, 1).get_matrix()
        ntools.assert_equal(a.shape, (2, 1))

        a = EigenArray(2, 2).get_matrix()
        ntools.assert_equal(a.shape, (2, 2))

        a = EigenArray(4, 4).get_matrix()
        ntools.assert_equal(a.shape, (4, 4))

    def test_dynamic_size_init(self):
        a = EigenArray(2, dynamic_rows=True).get_matrix()
        ntools.assert_equal(a.shape, (2, 1))

        a = EigenArray(300, dynamic_rows=True).get_matrix()
        ntools.assert_equal(a.shape, (300, 1))

        a = EigenArray(1234, 256, dynamic_rows=True, dynamic_cols=True).get_matrix()
        ntools.assert_equal(a.shape, (1234, 256))

    # Turn this test off while we're dealing only with dynamic shapes
    #def test_invalid_shape_init(self):
    #    ntools.assert_raises(
    #        VitalInvalidStaticEigenShape,
    #        EigenArray,
    #        5,
    #    )

    #    ntools.assert_raises(
    #        VitalInvalidStaticEigenShape,
    #        EigenArray,
    #        400, 500
    #    )

    def test_order_transform(self):
        a = EigenArray(2, 3)
        d = a.get_matrix()  # The data pointer
        # column-major 2x3 matrix [[ 1 2 3 ]  (Eigen format)
        #                          [ 4 5 6 ]]
        d[0][0] = 1; d[0][1] = 2; d[0][2] = 3
        d[1][0] = 4; d[1][1] = 5; d[1][2] = 6

        numpy.testing.assert_array_equal(a.get_matrix(), [[1., 2., 3.],
                                             [4., 5., 6.]])

        ntools.assert_equal(a.get_matrix()[0, 0], 1)
        ntools.assert_equal(a.get_matrix()[0, 1], 2)
        ntools.assert_equal(a.get_matrix()[0, 2], 3)
        ntools.assert_equal(a.get_matrix()[1, 0], 4)
        ntools.assert_equal(a.get_matrix()[1, 1], 5)
        ntools.assert_equal(a.get_matrix()[1, 2], 6)

    def test_mutability(self):
        a = EigenArray(2, 3)
        d = a.get_matrix()  # The data pointer
        for i in range(2):
          for j in range(3):
            d[i][j] = 0
        numpy.testing.assert_array_equal(a.get_matrix(), [[0, 0, 0],
                                             [0, 0, 0]])

        d[:] = 1
        numpy.testing.assert_array_equal(a.get_matrix(), [[1, 1, 1],
                                             [1, 1, 1]])

        d[1, 0] = 2
        numpy.testing.assert_array_equal(a.get_matrix(), [[1, 1, 1],
                                             [2, 1, 1]])

        d[:, 2] = 3
        numpy.testing.assert_array_equal(a.get_matrix(), [[1, 1, 3],
                                             [2, 1, 3]])

        d += 1
        numpy.testing.assert_array_equal(a.get_matrix(), [[2, 2, 4],
                                             [3, 2, 4]])

        b = d*0
        numpy.testing.assert_array_equal(a.get_matrix(), [[2, 2, 4],
                                             [3, 2, 4]])
        numpy.testing.assert_array_equal(b, [[0, 0, 0],
                                             [0, 0, 0]])

    def test_from_array(self):
        # from list
        expected_list = [[0.4, 0],
                         [1, 1.123],
                         [2.253, 4.768124]]
        ea = EigenArray.from_array(expected_list)
        em = ea.get_matrix()
        numpy.testing.assert_array_equal(em, expected_list)

        # from ndarray
        expected_ndar = numpy.array(expected_list)
        ea = EigenArray.from_array(expected_ndar)
        em = ea.get_matrix()
        numpy.testing.assert_array_equal(em, expected_ndar)

        # from EigenArray, which should return the input object
        ea = EigenArray(3, 2)
        em = ea.get_matrix()
        em[:] = expected_list
        ea2 = EigenArray.from_array(em)
        em2 = ea2.get_matrix()
        numpy.testing.assert_array_equal(em2, em)

    def test_from_array_1D(self):
        # 1-dim iterables/vectors are treated as column vectors
        input = [[1], [2], [3], [4]]
        expected = [[1],
                    [2],
                    [3],
                    [4]]

        e = EigenArray.from_array(input)
        em = e.get_matrix()
        numpy.testing.assert_equal(em, expected)

        e2 = EigenArray.from_array(em)
        em2 = e2.get_matrix()
        numpy.testing.assert_equal(em, em2)

    def test_norm(self):
        e = EigenArray.from_array([[1],[2],[3],[4]])
        numpy.linalg.norm(e.get_matrix()) == numpy.sqrt(1+4+9+16)
