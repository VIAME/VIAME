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

Tests for python Homography interface

"""
from __future__ import print_function
import sys
import os
import unittest

import nose.tools
import numpy

from kwiver.vital.exceptions.math import PointMapsToInfinityException
from kwiver.vital.types import (
    EigenArray,
    Homography,
)


class TestHomography (unittest.TestCase):

    def test_ident_init(self):
        h_d = Homography()
        h_f = Homography('f')

    def test_matrix_init(self):
        # Test that construction does not fail when passing valid matrices as
        # initializer
        m1 = [[0,     1,   3],
              [0.3, 0.1,  10],
              [-1,  8.1, 4.7]]
        m2_d = EigenArray.from_array(m1, 'd')
        m2_f = EigenArray.from_array(m1, 'f')

        Homography.from_matrix(m1, 'd')
        Homography.from_matrix(m1, 'f')
        Homography.from_matrix(m2_d.get_matrix(), 'd')
        Homography.from_matrix(m2_d.get_matrix(), 'f')
        Homography.from_matrix(m2_f.get_matrix(), 'd')
        Homography.from_matrix(m2_f.get_matrix(), 'f')

    def test_random(self):
        Homography.random('d')
        Homography.random('f')

    def test_typename(self):
        h_d = Homography('d')
        h_f = Homography('f')

        nose.tools.assert_equal(h_d.type_name, 'd')
        nose.tools.assert_equal(h_f.type_name, 'f')

    def test_as_matrix(self):
        numpy.testing.assert_almost_equal(
            Homography('d').as_matrix(),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )
        numpy.testing.assert_almost_equal(
            Homography('f').as_matrix(),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )

        m = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        numpy.testing.assert_almost_equal(
            Homography.from_matrix(m, 'd').as_matrix(),
            m
        )
        numpy.testing.assert_almost_equal(
            Homography.from_matrix(m, 'f').as_matrix(),
            m
        )

        m = [[.1, .2, .3],
             [.4, .5, .6],
             [.7, .8, .9]]
        numpy.testing.assert_almost_equal(
            Homography.from_matrix(m, 'd').as_matrix(),
            m
        )
        numpy.testing.assert_almost_equal(
            Homography.from_matrix(m, 'f').as_matrix(),
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

    def test_numeric_invertibility(self):
        exp_result = Homography.from_matrix([[-.5, -2.5, 1.5],
                                             [-1.5, 1.5, -.5],
                                             [1.5,   .5, -.5]])

        h = Homography.from_matrix([[1, 1, 2],
                                    [3, 4, 5],
                                    [6, 7, 9]])
        h_inv = h.inverse()
        numpy.testing.assert_array_equal(h_inv, exp_result.as_matrix())

        h = Homography.from_matrix([[1, 1, 2],
                                    [3, 4, 5],
                                    [6, 7, 9]],
                                   'f')
        h_inv = h.inverse()
        numpy.testing.assert_array_equal(h_inv, exp_result.as_matrix())

    def test_normalize(self):
        h = Homography.from_matrix([[-.5, -2.5, 1.5],
                                    [-1.5, 1.5, -.5],
                                    [1.5,   .5, -.5]])
        e = Homography.from_matrix([[1,   5, -3],
                                    [3,  -3,  1],
                                    [-3, -1,  1]])
        numpy.testing.assert_array_equal(h.normalize(), e.as_matrix())

    def test_point_map(self):
        h_f = Homography('f')
        h_d = Homography('d')

        p_af = EigenArray.from_array([[2.2, 3.3]], 'f')
        p_f = p_af.get_matrix()[0]
        p_ad = EigenArray.from_array([[5.5, 6.6]], 'd')
        p_d = p_ad.get_matrix()[0]

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

        # Code to generate truth
        h = numpy.random.rand(3,3)
        h = h/numpy.linalg.norm(h)
        p0 = numpy.random.rand(3); p0[2] = 1
        p1 = numpy.dot(h, p0)
        p1 = p1[:2]/p1[2]
        h_d = Homography.from_matrix(h, 'd')

        # map from Numpy array.
        numpy.testing.assert_almost_equal(
            h_d.map(p0[:2]).ravel(), p1
        )

        # map from EigenArray
        p0 = EigenArray.from_array([p0[:2]])
        numpy.testing.assert_almost_equal(
            h_d.map(p0.get_matrix()[0]).ravel(), p1
        )

        # Another explicit case.
        p0 = numpy.array([1923.47,645.676,1])
        h = numpy.array([[5.491496261770000276e-01,-1.125428185150000038e-01,
                          1.358427031619999923e+02],
                         [-1.429513389049999993e-02	,6.035527375529999849e-01,
                          5.923971959490000216e+01],
                         [-2.042570000000000164e-06,-2.871670000000000197e-07,
                          1]])
        p1 = numpy.dot(h, p0);      p1 = p1[:2]/p1[2]
        H = Homography.from_matrix(h)
        P = EigenArray.from_array([p0[:2]])
        numpy.testing.assert_almost_equal(
            H.map(P.get_matrix()[0]).ravel(), p1
        )


    def test_point_map_zero_div(self):
        test_p = [1, 1]

        for dtype, e in [['f', numpy.finfo('f').min],
                         ['d', sys.float_info.min]]:
            print("Dtype:", dtype)
            print("E:", e)

            # where [2,2] = 0
            h = Homography.from_matrix([[1, 0, 1],
                                        [0, 1, 1],
                                        [0, 0, 0]],
                                       dtype)
            nose.tools.assert_raises(
                RuntimeError,
                h.map, test_p
            )

            # Where [2,2] = e which is approximately 0
            e = sys.float_info.min
            h = Homography.from_matrix([[1, 0, 1],
                                        [0, 1, 1],
                                        [0, 0, e]],
                                       dtype)
            print("E Matrix:", h.as_matrix())
            nose.tools.assert_raises(
                RuntimeError,
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
