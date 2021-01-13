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

Tests for python Homography interface

"""
from __future__ import print_function
import sys
import os
import unittest

import nose.tools
import numpy as np

from kwiver.vital.exceptions.math import PointMapsToInfinityException
from kwiver.vital.types.homography import *


class TestHomography (unittest.TestCase):
    def test_no_init_base(self):
         with nose.tools.assert_raises_regexp(
            TypeError, "kwiver.vital.types.homography.BaseHomography: No constructor defined!"
        ):
            BaseHomography()

    def test_ident_init(self):
        h_d = HomographyD()
        h_f = HomographyF()

    def test_matrix_init(self):
        # Test that construction does not fail when passing valid matrices as
        # initializer
        m1 = [[0,     1,   3],
              [0.3, 0.1,  10],
              [-1,  8.1, 4.7]]
        m2_d = np.array(m1, dtype = np.float64)
        m2_f = np.array(m1, dtype = np.float32)

        HomographyD(m1)
        HomographyF(m1)
        HomographyD(m2_d)
        HomographyF(m2_d)
        HomographyD(m2_f)
        HomographyF(m2_f)

    def test_random(self):
        HomographyD.random()
        HomographyF.random()

    def test_typename(self):
        h_d = HomographyD()
        h_f = HomographyF()

        nose.tools.assert_equal(h_d.type_name, 'd')
        nose.tools.assert_equal(h_f.type_name, 'f')

    def test_as_matrix(self):
        np.testing.assert_almost_equal(
            HomographyD().matrix(),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )
        np.testing.assert_almost_equal(
            HomographyF().matrix(),
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )

        m = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        np.testing.assert_almost_equal(
            HomographyD(m).matrix(),
            m
        )
        np.testing.assert_almost_equal(
            HomographyF(m).matrix(),
            m
        )

        m = [[.1, .2, .3],
             [.4, .5, .6],
             [.7, .8, .9]]
        np.testing.assert_almost_equal(
            HomographyD(m).matrix(),
            m
        )
        np.testing.assert_almost_equal(
            HomographyF(m).matrix(),
            m
        )

    def test_numeric_invertibility(self):
        exp_result = HomographyD([[-.5, -2.5, 1.5],
                                  [-1.5, 1.5, -.5],
                                  [1.5,   .5, -.5]])

        h = HomographyD([[1, 1, 2],
                         [3, 4, 5],
                         [6, 7, 9]])
        h_inv = h.inverse().matrix()
        np.testing.assert_array_equal(h_inv, exp_result.matrix())

        h = HomographyF([[1, 1, 2],
                         [3, 4, 5],
                         [6, 7, 9]])
        h_inv = h.inverse().matrix()
        np.testing.assert_array_equal(h_inv, exp_result.matrix())

    def test_normalize(self):
        h = HomographyD([[-.5, -2.5, 1.5],
                         [-1.5, 1.5, -.5],
                         [1.5,   .5, -.5]])
        e = HomographyD([[1,   5, -3],
                         [3,  -3,  1],
                         [-3, -1,  1]])
        np.testing.assert_array_equal(h.normalize().matrix(), e.matrix())

    def test_point_map(self):
        h_f = HomographyF()
        h_d = HomographyD()

        p_af = np.array([[2.2, 3.3]], dtype = np.float32)
        p_f = p_af[0]
        p_ad = np.array([[5.5, 6.6]], dtype = np.float64)
        p_d = p_ad[0]

        # float-float
        np.testing.assert_almost_equal(
            h_f.map(p_f), p_f
        )
        # float-double
        np.testing.assert_almost_equal(
            h_f.map(p_d), p_d
        )
        # double-float
        np.testing.assert_almost_equal(
            h_d.map(p_f), p_f
        )
        # double-double
        np.testing.assert_almost_equal(
            h_d.map(p_d), p_d
        )

        # Code to generate truth
        h = np.random.rand(3,3)
        h = h/np.linalg.norm(h)
        p0 = np.random.rand(3); p0[2] = 1
        p1 = np.dot(h, p0)
        p1 = p1[:2]/p1[2]
        h_d = HomographyD(h)

        # map from np array.
        np.testing.assert_almost_equal(
            h_d.map(p0[:2]).ravel(), p1
        )

        # Another explicit case.
        p0 = np.array([1923.47,645.676,1])
        h = np.array([[5.491496261770000276e-01,-1.125428185150000038e-01,
                          1.358427031619999923e+02],
                         [-1.429513389049999993e-02	,6.035527375529999849e-01,
                          5.923971959490000216e+01],
                         [-2.042570000000000164e-06,-2.871670000000000197e-07,
                          1]])
        p1 = np.dot(h, p0);      p1 = p1[:2]/p1[2]
        H = HomographyD(h)
        P = np.array([p0[:2]])
        np.testing.assert_almost_equal(
            H.map(P[0]).ravel(), p1
        )


    def test_point_map_zero_div(self):
        test_p = [1, 1]

        for ctor, dtype, e in [[HomographyF, 'f', np.finfo('f').min],
                         [HomographyD, 'd', sys.float_info.min]]:
            print("Dtype:", dtype)
            print("E:", e)

            # where [2,2] = 0
            h = ctor([[1, 0, 1],
                      [0, 1, 1],
                      [0, 0, 0]])
            nose.tools.assert_raises(
                RuntimeError,
                h.map, test_p
            )

            # Where [2,2] = e which is approximately 0
            e = sys.float_info.min
            h = ctor([[1, 0, 1],
                      [0, 1, 1],
                      [0, 0, e]])
            print("E Matrix:", h.matrix())
            nose.tools.assert_raises(
                RuntimeError,
                h.map, test_p
            )

            # Where [2,2] = 0.5, which should be valid
            h = ctor([[1, 0,  1],
                      [0, 1,  1],
                      [0, 0, .5]])
            r = h.map(test_p)
            nose.tools.assert_almost_equal(r[0], 4)
            nose.tools.assert_almost_equal(r[1], 4)

    def test_multiply(self):
        # Test multiplying homographies together
        h_ident = HomographyD()
        h_valued = HomographyD([[1, 0,  1],
                                [0, 1,  1],
                                [0, 0, .5]])

        r1 = h_ident * h_ident
        np.testing.assert_array_almost_equal(h_ident.matrix(), r1.matrix())

        r2 = h_ident * h_valued
        np.testing.assert_array_almost_equal(r2.matrix(), h_valued.matrix())

        r3 = h_valued * h_ident
        np.testing.assert_array_almost_equal(r3.matrix(), h_valued.matrix())
