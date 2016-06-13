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

Test Python interface to Eigen::Matrix wrapping via Numpy ndarray sub-class

"""

import unittest

import nose.tools as ntools
import numpy

from vital.exceptions.eigen import VitalInvalidStaticEigenShape
from vital.types import EigenArray


class TestVitalEigenMatrix (unittest.TestCase):

    def test_valid_static_size_init(self):
        """
        Test construction of some of the static sizes.
        """
        a = EigenArray() # default shape
        ntools.assert_equal(a.shape, (2, 1))

        a = EigenArray(2, 1)
        ntools.assert_equal(a.shape, (2, 1))

        a = EigenArray(2, 2)
        ntools.assert_equal(a.shape, (2, 2))

        a = EigenArray(4, 4)
        ntools.assert_equal(a.shape, (4, 4))

    def test_dynamic_size_init(self):
        a = EigenArray(2, dynamic_rows=True)
        ntools.assert_equal(a.shape, (2, 1))

        a = EigenArray(300, dynamic_rows=True)
        ntools.assert_equal(a.shape, (300, 1))

        a = EigenArray(1234, 256, dynamic_rows=True, dynamic_cols=True)
        ntools.assert_equal(a.shape, (1234, 256))

    def test_invalid_shape_init(self):
        ntools.assert_raises(
            VitalInvalidStaticEigenShape,
            EigenArray,
            5,
        )

        ntools.assert_raises(
            VitalInvalidStaticEigenShape,
            EigenArray,
            400, 500
        )

    def test_order_transform(self):
        a = EigenArray(2, 3)
        d = a.base.base  # The data pointer
        # column-major 2x3 matrix [[ 1 2 3 ]  (Eigen format)
        #                          [ 4 5 6 ]]
        d[0] = 1; d[2] = 2; d[4] = 3
        d[1] = 4; d[3] = 5; d[5] = 6

        numpy.testing.assert_array_equal(a, [[1., 2., 3.],
                                             [4., 5., 6.]])

        ntools.assert_equal(a.at_eigen_base_index(0, 0), 1)
        ntools.assert_equal(a.at_eigen_base_index(0, 1), 2)
        ntools.assert_equal(a.at_eigen_base_index(0, 2), 3)
        ntools.assert_equal(a.at_eigen_base_index(1, 0), 4)
        ntools.assert_equal(a.at_eigen_base_index(1, 1), 5)
        ntools.assert_equal(a.at_eigen_base_index(1, 2), 6)

    def test_mutability(self):
        a = EigenArray(2, 3)
        d = a.base.base  # The data pointer
        for i in xrange(6):
            d[i] = 0
        numpy.testing.assert_array_equal(a, [[0, 0, 0],
                                             [0, 0, 0]])

        a[:] = 1
        numpy.testing.assert_array_equal(a, [[1, 1, 1],
                                             [1, 1, 1]])

        a[1, 0] = 2
        numpy.testing.assert_array_equal(a, [[1, 1, 1],
                                             [2, 1, 1]])

        a[:, 2] = 3
        numpy.testing.assert_array_equal(a, [[1, 1, 3],
                                             [2, 1, 3]])

        a += 1
        numpy.testing.assert_array_equal(a, [[2, 2, 4],
                                             [3, 2, 4]])

        b = a*0
        numpy.testing.assert_array_equal(a, [[2, 2, 4],
                                             [3, 2, 4]])
        numpy.testing.assert_array_equal(b, [[0, 0, 0],
                                             [0, 0, 0]])

    def test_from_iterable(self):
        # from list
        expected_list = [[0.4, 0],
                         [1, 1.123],
                         [2.253, 4.768124]]
        ea = EigenArray.from_iterable(expected_list, target_shape=(3, 2))
        numpy.testing.assert_array_equal(ea, expected_list)

        # from ndarray
        expected_ndar = numpy.array(expected_list)
        ea = EigenArray.from_iterable(expected_ndar, target_shape=(3, 2))
        numpy.testing.assert_array_equal(ea, expected_ndar)

        # from EigenArray, which should return the input object
        ea = EigenArray(3, 2)
        ea[:] = expected_list
        ea2 = EigenArray.from_iterable(ea, target_shape=(3, 2))
        numpy.testing.assert_array_equal(ea2, ea)
        ntools.assert_is(ea, ea2)
        ntools.assert_true(ea is ea2)

    def test_from_iterable_1D(self):
        # 1-dim iterables/vectors are treated as column vectors
        input = [1, 2, 3, 4]
        expected = [[1],
                    [2],
                    [3],
                    [4]]

        e = EigenArray.from_iterable(input)
        numpy.testing.assert_equal(e, expected)

        e2 = EigenArray.from_iterable(e)
        numpy.testing.assert_equal(e, e2)
