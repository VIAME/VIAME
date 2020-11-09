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

Tests for Vital python Covariance class

"""
from __future__ import print_function
import unittest

import nose.tools
import numpy as np

from kwiver.vital.types.covariance import *


class TestVitalCovariance(unittest.TestCase):
    def test_new_identity(self):
        # Valid dimensions and types
        c = Covar2d()
        print("constructed matrix:\n", c.matrix())
        c = Covar3d()
        print("constructed matrix:\n", c.matrix())
        c = Covar4d()
        print("constructed matrix:\n", c.matrix())
        c = Covar2f()
        print("constructed matrix:\n", c.matrix())
        c = Covar3f()
        print("constructed matrix:\n", c.matrix())
        c = Covar4f()
        print("constructed matrix:\n", c.matrix())

    def test_new_scalar(self):
        c = Covar2d(2.0)
        print("constructed matrix:\n", c.matrix())
        c = Covar3d(2.0)
        print("constructed matrix:\n", c.matrix())
        c = Covar4d(2.0)
        print("constructed matrix:\n", c.matrix())
        c = Covar2f(2.0)
        print("constructed matrix:\n", c.matrix())
        c = Covar3f(2.0)
        print("constructed matrix:\n", c.matrix())
        c = Covar4f(2.0)
        print("constructed matrix:\n", c.matrix())

        c = Covar2d(14.675)
        print("constructed matrix:\n", c.matrix())
        c = Covar3d(14.675)
        print("constructed matrix:\n", c.matrix())
        c = Covar4d(14.675)
        print("constructed matrix:\n", c.matrix())
        c = Covar2f(14.675)
        print("constructed matrix:\n", c.matrix())
        c = Covar3f(14.675)
        print("constructed matrix:\n", c.matrix())
        c = Covar4f(14.675)
        print("constructed matrix:\n", c.matrix())

    def test_new_matrix(self):
        m = np.array([[1, 1], [1, 1]])
        c = Covar2d(m)
        m_out = c.matrix()
        print("input matrix:\n", m)
        print("output matrix:\n", m_out)
        np.testing.assert_array_equal(m_out, m)

        # Type casting should be handled
        m = np.array([[1, 1], [1, 1]], dtype=np.float32)
        c = Covar2d(m)
        m_out = c.matrix()
        print("input matrix:\n", m)
        print("output matrix:\n", m_out)
        np.testing.assert_array_equal(m_out, m)

        # Any other numpy array of the correct shape should be acceptable
        m = np.ndarray((2, 2))
        m[:] = 3.0
        c = Covar2f(m)
        m_out = c.matrix()
        print("input matrix:\n", m)
        print("output matrix:\n", m_out)
        np.testing.assert_array_equal(m_out, m)

        # Diagonally congruent values should be averages when initializing with
        # matrix
        m = np.eye(3, dtype=np.double)
        m[0, 2] = 2.0
        m_expected = m.copy()
        m_expected[0, 2] = 1.0
        m_expected[2, 0] = 1.0
        c = Covar3d(m)
        m_out = c.matrix()
        print("input matrix:\n", m)
        print("output matrix:\n", m_out)
        np.testing.assert_array_equal(m_out, m_expected)

    def test_get_value(self):
        m = np.ndarray((4, 4))
        # [[ 0  2  4  6  ]                [[ 0  5  10 15 ]
        #  [ 8  10 12 14 ]  -> should ->   [ 5  10 15 20 ]
        #  [ 16 18 20 22 ]                 [ 10 15 20 25 ]
        #  [ 24 26 28 30 ]]                [ 15 20 25 30 ]]
        m.reshape((16,))[:] = list(range(0, 32, 2))

        c = Covar4d(m)
        # Test matrix upper triangle locations
        nose.tools.assert_equal(c[0, 0], 0)
        nose.tools.assert_equal(c[0, 1], 5)
        nose.tools.assert_equal(c[0, 2], 10)
        nose.tools.assert_equal(c[0, 3], 15)
        nose.tools.assert_equal(c[1, 1], 10)
        nose.tools.assert_equal(c[1, 2], 15)
        nose.tools.assert_equal(c[1, 3], 20)
        nose.tools.assert_equal(c[2, 2], 20)
        nose.tools.assert_equal(c[2, 3], 25)
        nose.tools.assert_equal(c[3, 3], 30)

        nose.tools.assert_equal(c[0, 1], c[1, 0])
        nose.tools.assert_equal(c[0, 2], c[2, 0])
        nose.tools.assert_equal(c[0, 3], c[3, 0])
        nose.tools.assert_equal(c[1, 2], c[2, 1])
        nose.tools.assert_equal(c[1, 3], c[3, 1])
        nose.tools.assert_equal(c[2, 3], c[3, 2])

        c = Covar4f(m)
        # Test matrix upper triangle locations
        nose.tools.assert_equal(c[0, 0], 0)
        nose.tools.assert_equal(c[0, 1], 5)
        nose.tools.assert_equal(c[0, 2], 10)
        nose.tools.assert_equal(c[0, 3], 15)
        nose.tools.assert_equal(c[1, 1], 10)
        nose.tools.assert_equal(c[1, 2], 15)
        nose.tools.assert_equal(c[1, 3], 20)
        nose.tools.assert_equal(c[2, 2], 20)
        nose.tools.assert_equal(c[2, 3], 25)
        nose.tools.assert_equal(c[3, 3], 30)

        nose.tools.assert_equal(c[0, 1], c[1, 0])
        nose.tools.assert_equal(c[0, 2], c[2, 0])
        nose.tools.assert_equal(c[0, 3], c[3, 0])
        nose.tools.assert_equal(c[1, 2], c[2, 1])
        nose.tools.assert_equal(c[1, 3], c[3, 1])
        nose.tools.assert_equal(c[2, 3], c[3, 2])

    def test_get_oob(self):
        # 2x2 covariance mat
        c = Covar2d()
        _ = c[0, 0]  # Valid access
        nose.tools.assert_raises(IndexError, c.__getitem__, (0, 2))
        nose.tools.assert_raises(IndexError, c.__getitem__, (-1, 0))

        c = Covar2f()
        _ = c[0, 0]  # Valid access
        nose.tools.assert_raises(IndexError, c.__getitem__, (0, 2))
        nose.tools.assert_raises(IndexError, c.__getitem__, (-1, 0))

    def test_set(self):
        m = np.ndarray((4, 4))
        # [[ 0  2  4  6  ]                [[ 0  5  10 15 ]
        #  [ 8  10 12 14 ]  -> should ->   [ 5  10 15 20 ]
        #  [ 16 18 20 22 ]                 [ 10 15 20 25 ]
        #  [ 24 26 28 30 ]]                [ 15 20 25 30 ]]
        m.reshape((16,))[:] = list(range(0, 32, 2))

        c = Covar4d(m)

        # modify some locations
        c[0, 1] = 1
        c[2, 2] = 3

        # Test matrix upper triangle locations
        nose.tools.assert_equal(c[0, 0], 0)
        nose.tools.assert_equal(c[0, 1], 1)
        nose.tools.assert_equal(c[0, 2], 10)
        nose.tools.assert_equal(c[0, 3], 15)
        nose.tools.assert_equal(c[1, 1], 10)
        nose.tools.assert_equal(c[1, 2], 15)
        nose.tools.assert_equal(c[1, 3], 20)
        nose.tools.assert_equal(c[2, 2], 3)
        nose.tools.assert_equal(c[2, 3], 25)
        nose.tools.assert_equal(c[3, 3], 30)

        nose.tools.assert_equal(c[0, 1], c[1, 0])
        nose.tools.assert_equal(c[0, 2], c[2, 0])
        nose.tools.assert_equal(c[0, 3], c[3, 0])
        nose.tools.assert_equal(c[1, 2], c[2, 1])
        nose.tools.assert_equal(c[1, 3], c[3, 1])
        nose.tools.assert_equal(c[2, 3], c[3, 2])

        # Set in upper triangle and see it reflect in lower
        c[0, 2] = 42
        nose.tools.assert_equal(c[2, 0], 42)

        # Change something in lower triangle and see it reflected in upper
        c[2, 1] = 43
        nose.tools.assert_equal(c[1, 2], 43)

        # FLOAT
        c = Covar4f(m)
        # modify some locations
        c[0, 1] = 1
        c[2, 2] = 3

        # Test matrix upper triangle locations
        nose.tools.assert_equal(c[0, 0], 0)
        nose.tools.assert_equal(c[0, 1], 1)
        nose.tools.assert_equal(c[0, 2], 10)
        nose.tools.assert_equal(c[0, 3], 15)
        nose.tools.assert_equal(c[1, 1], 10)
        nose.tools.assert_equal(c[1, 2], 15)
        nose.tools.assert_equal(c[1, 3], 20)
        nose.tools.assert_equal(c[2, 2], 3)
        nose.tools.assert_equal(c[2, 3], 25)
        nose.tools.assert_equal(c[3, 3], 30)

        nose.tools.assert_equal(c[0, 1], c[1, 0])
        nose.tools.assert_equal(c[0, 2], c[2, 0])
        nose.tools.assert_equal(c[0, 3], c[3, 0])
        nose.tools.assert_equal(c[1, 2], c[2, 1])
        nose.tools.assert_equal(c[1, 3], c[3, 1])
        nose.tools.assert_equal(c[2, 3], c[3, 2])

        # Set in upper triangle and see it reflect in lower
        c[0, 2] = 42
        nose.tools.assert_equal(c[2, 0], 42)

        # Change something in lower triangle and see it reflected in upper
        c[2, 1] = 43
        nose.tools.assert_equal(c[1, 2], 43)

    def test_set_oob(self):
        # 2x2 covariance mat
        c = Covar2f()
        c[0, 0] = 1  # Valid set
        nose.tools.assert_raises(IndexError, c.__setitem__, (0, 2), 1)
        nose.tools.assert_raises(IndexError, c.__setitem__, (-1, 0), 1)

        c = Covar2d()
        c[0, 0] = 1  # Valid set
        nose.tools.assert_raises(IndexError, c.__setitem__, (0, 2), 1)
        nose.tools.assert_raises(IndexError, c.__setitem__, (-1, 0), 1)
