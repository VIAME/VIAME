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

Tests for python Homography interface

"""
import ctypes
import sys
import unittest

import nose.tools
import numpy

from vital.exceptions.math import PointMapsToInfinityException
from vital.types import (
    EigenArray,
    Homography,
)


class TestHomography (unittest.TestCase):

    def test_ident_init(self):
        h_d = Homography()
        h_f = Homography(ctypes.c_float)

    def test_matrix_init(self):
        # Test that construction does not fail when passing valid matrices as
        # initializer
        m1 = [[0,     1,   3],
              [0.3, 0.1,  10],
              [-1,  8.1, 4.7]]
        m2_d = EigenArray.from_iterable(m1, ctypes.c_double, (3, 3))
        m2_f = EigenArray.from_iterable(m1, ctypes.c_float, (3, 3))

        Homography.from_matrix(m1, ctypes.c_double)
        Homography.from_matrix(m1, ctypes.c_float)
        Homography.from_matrix(m2_d, ctypes.c_double)
        Homography.from_matrix(m2_d, ctypes.c_float)
        Homography.from_matrix(m2_f, ctypes.c_double)
        Homography.from_matrix(m2_f, ctypes.c_float)

    def test_typename(self):
        h_d = Homography(ctypes.c_double)
        h_f = Homography(ctypes.c_float)

        nose.tools.assert_equal(h_d.type_name, 'd')
        nose.tools.assert_equal(h_f.type_name, 'f')

    def test_as_matrix(self):
        numpy.testing.assert_almost_equal(
            Homography(ctypes.c_double).as_matrix(),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )
        numpy.testing.assert_almost_equal(
            Homography(ctypes.c_float).as_matrix(),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )

        m = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        numpy.testing.assert_almost_equal(
            Homography.from_matrix(m, ctypes.c_double).as_matrix(),
            m
        )
        numpy.testing.assert_almost_equal(
            Homography.from_matrix(m, ctypes.c_float).as_matrix(),
            m
        )

        m = [[.1, .2, .3],
             [.4, .5, .6],
             [.7, .8, .9]]
        numpy.testing.assert_almost_equal(
            Homography.from_matrix(m, ctypes.c_double).as_matrix(),
            m
        )
        numpy.testing.assert_almost_equal(
            Homography.from_matrix(m, ctypes.c_float).as_matrix(),
            m
        )

    def test_equal(self):
        # Identity should be equal to itself
        h1 = Homography()
        h2 = Homography()
        nose.tools.assert_equal(h1, h2)

        # manually constructed homographies should be equal
        m = [[.1, .2, .3],
             [.4, .5, .6],
             [.7, .8, .9]]
        h1 = Homography.from_matrix(m)
        h2 = Homography.from_matrix(m)
        nose.tools.assert_equal(h1, h2)
        # and should also be different than identity
        nose.tools.assert_not_equal(h1, Homography())
        nose.tools.assert_not_equal(h2, Homography())

        # Should not be equal to these
        nose.tools.assert_not_equal(h1, 0)
        nose.tools.assert_not_equal(h2, 'foo')

    def test_clone(self):
        m = [[.1, .2, .3],
             [.4, .5, .6],
             [.7, .8, .9]]
        # Cloning a matrix should yield a new instance that has the same value
        # as the original
        h1 = Homography.from_matrix(m, ctypes.c_double)
        h2 = h1.clone()
        nose.tools.assert_false(h1 is h2)
        nose.tools.assert_not_equal(ctypes.addressof(h1.c_pointer.contents),
                                    ctypes.addressof(h2.c_pointer.contents))
        numpy.testing.assert_almost_equal(h1.as_matrix(), h2.as_matrix())

        # Cloning should carry over data type
        nose.tools.assert_equal(h1.type_name, h2.type_name)

        h3 = Homography.from_matrix(m, ctypes.c_float)
        h4 = h3.clone()
        nose.tools.assert_equal(h3.type_name, h4.type_name)

    def test_numeric_invertibility(self):
        exp_result = Homography.from_matrix([[-.5, -2.5, 1.5],
                                             [-1.5, 1.5, -.5],
                                             [1.5,   .5, -.5]])

        h = Homography.from_matrix([[1, 1, 2],
                                    [3, 4, 5],
                                    [6, 7, 9]])
        h_inv = h.inverse()
        nose.tools.assert_equal(h_inv, exp_result)

        h = Homography.from_matrix([[1, 1, 2],
                                    [3, 4, 5],
                                    [6, 7, 9]],
                                   ctypes.c_float)
        h_inv = h.inverse()
        nose.tools.assert_equal(h_inv, exp_result)

    def test_normalize(self):
        h = Homography.from_matrix([[-.5, -2.5, 1.5],
                                    [-1.5, 1.5, -.5],
                                    [1.5,   .5, -.5]])
        e = Homography.from_matrix([[1,   5, -3],
                                    [3,  -3,  1],
                                    [-3, -1,  1]])
        nose.tools.assert_equal(h.normalize(), e)

    def test_point_map(self):
        h_f = Homography(ctypes.c_float)
        h_d = Homography(ctypes.c_double)

        p_f = EigenArray.from_iterable([2.2, 3.3], ctypes.c_float)
        p_d = EigenArray.from_iterable([5.5, 6.6], ctypes.c_double)

        # float-float
        numpy.testing.assert_almost_equal(
            h_f.map(p_f), p_f
        )
        # float-double
        numpy.testing.assert_almost_equal(
            h_f.map(p_d), p_d
        )
        # double-float
        numpy.testing.assert_almost_equal(
            h_d.map(p_f), p_f
        )
        # double-double
        numpy.testing.assert_almost_equal(
            h_d.map(p_d), p_d
        )

    def test_point_map_zero_div(self):
        test_p = [1, 1]

        for dtype, e in [[ctypes.c_float, numpy.finfo(ctypes.c_float).min],
                         [ctypes.c_double, sys.float_info.min]]:
            print "Dtype:", dtype
            print "E:", e

            # where [2,2] = 0
            h = Homography.from_matrix([[1, 0, 1],
                                        [0, 1, 1],
                                        [0, 0, 0]],
                                       dtype)
            nose.tools.assert_raises(
                PointMapsToInfinityException,
                h.map, test_p
            )

            # Where [2,2] = e which is approximately 0
            e = sys.float_info.min
            h = Homography.from_matrix([[1, 0, 1],
                                        [0, 1, 1],
                                        [0, 0, e]],
                                       dtype)
            print "E Matrix:", h.as_matrix()
            nose.tools.assert_raises(
                PointMapsToInfinityException,
                h.map, test_p
            )

            # Where [2,2] = 0.5, which should be valid
            h = Homography.from_matrix([[1, 0,  1],
                                        [0, 1,  1],
                                        [0, 0, .5]],
                                       dtype)
            r = h.map(test_p)
            nose.tools.assert_almost_equal(r[0], 4)
            nose.tools.assert_almost_equal(r[1], 4)

    def test_multiply(self):
        # Test multiplying homographies together
        h_ident = Homography()
        h_valued = Homography.from_matrix([[1, 0,  1],
                                           [0, 1,  1],
                                           [0, 0, .5]])

        r1 = h_ident * h_ident
        nose.tools.assert_equal(h_ident, r1)

        r2 = h_ident * h_valued
        nose.tools.assert_equal(r2, h_valued)

        r3 = h_valued * h_ident
        nose.tools.assert_equal(r3, h_valued)

        # Failure, multiplying against non-homography
        def h_mult(h, v):
            return h * v

        nose.tools.assert_raises(
            ValueError,
            h_mult, h_ident, 1
        )
        nose.tools.assert_raises(
            ValueError,
            h_mult, h_ident, 'a string'
        )
