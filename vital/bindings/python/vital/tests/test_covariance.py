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

Tests for Vital python Covariance class

"""
import ctypes
import unittest

import nose.tools
import numpy

from vital.types import Covariance, EigenArray
from vital.util import VitalObject, VitalErrorHandle


class TestVitalCovariance (unittest.TestCase):

    def test_new_identity(self):
        # Valid dimensions and types
        c = Covariance(2, ctypes.c_double)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(3, ctypes.c_double)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(2, ctypes.c_float)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(3, ctypes.c_float)
        print 'constructed matrix:\n', c.to_matrix()

    def test_new_scalar(self):
        c = Covariance(2, ctypes.c_double, 2.)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(3, ctypes.c_double, 2.)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(2, ctypes.c_float, 2.)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(3, ctypes.c_float, 2.)
        print 'constructed matrix:\n', c.to_matrix()

        c = Covariance(2, ctypes.c_double, 14.675)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(3, ctypes.c_double, 14.675)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(2, ctypes.c_float, 14.675)
        print 'constructed matrix:\n', c.to_matrix()
        c = Covariance(3, ctypes.c_float, 14.675)
        print 'constructed matrix:\n', c.to_matrix()

    def test_new_matrix(self):
        m = EigenArray(2, 2, dtype=numpy.double)
        m[:] = 1.
        c = Covariance(2, ctypes.c_double, m)
        m_out = c.to_matrix()
        print 'input matrix:\n', m
        print 'output matrix:\n', m_out
        numpy.testing.assert_array_equal(m_out, m)

        # Type casting should be handled
        m = EigenArray(2, 2, dtype=numpy.float32)
        m[:] = 1.
        c = Covariance(2, ctypes.c_double, m)
        m_out = c.to_matrix()
        print 'input matrix:\n', m
        print 'output matrix:\n', m_out
        numpy.testing.assert_array_equal(m_out, m)

        # Any other numpy array of the correct shape should be acceptable
        m = numpy.ndarray((2, 2))
        m[:] = 3.
        c = Covariance(2, ctypes.c_float, init_scalar_or_matrix=m)
        m_out = c.to_matrix()
        print 'input matrix:\n', m
        print 'output matrix:\n', m_out
        numpy.testing.assert_array_equal(m_out, m)

        # Diagonally congruent values should be averages when initializing with
        # matrix
        m = numpy.eye(3, dtype=numpy.double)
        m[0,2] = 2.
        m_expected = m.copy()
        m_expected[0,2] = 1.
        m_expected[2,0] = 1.
        c = Covariance(3, init_scalar_or_matrix=m)
        m_out = c.to_matrix()
        print 'input matrix:\n', m
        print 'output matrix:\n', m_out
        numpy.testing.assert_array_equal(m_out, m_expected)

    def test_get_value(self):
        m = numpy.ndarray((3, 3))
        # [[ 0 1 2 ]               [[ 0 2 4 ]
        #  [ 3 4 5 ]  -> should ->  [ 2 4 6 ]
        #  [ 6 7 8 ]]               [ 4 6 8 ]]
        m.reshape((9,))[:] = range(9)

        c = Covariance(3, c_type=ctypes.c_double, init_scalar_or_matrix=m)
        # Test matrix upper triangle locations
        nose.tools.assert_equal(c[0,0], 0)
        nose.tools.assert_equal(c[0,1], 2)
        nose.tools.assert_equal(c[0,2], 4)
        nose.tools.assert_equal(c[1,1], 4)
        nose.tools.assert_equal(c[1,2], 6)
        nose.tools.assert_equal(c[2,2], 8)
        nose.tools.assert_equal(c[0,1], c[1,0])
        nose.tools.assert_equal(c[0,2], c[2,0])
        nose.tools.assert_equal(c[1,2], c[2,1])

        c = Covariance(3, c_type=ctypes.c_float, init_scalar_or_matrix=m)
        # Test matrix upper triangle locations
        nose.tools.assert_equal(c[0,0], 0)
        nose.tools.assert_equal(c[0,1], 2)
        nose.tools.assert_equal(c[0,2], 4)
        nose.tools.assert_equal(c[1,1], 4)
        nose.tools.assert_equal(c[1,2], 6)
        nose.tools.assert_equal(c[2,2], 8)
        nose.tools.assert_equal(c[0,1], c[1,0])
        nose.tools.assert_equal(c[0,2], c[2,0])
        nose.tools.assert_equal(c[1,2], c[2,1])

    def test_get_oob(self):
        # 2x2 covariance mat
        c = Covariance(c_type=ctypes.c_double)
        _ = c[0, 0]  # Valid access
        nose.tools.assert_raises(
            IndexError,
            c.__getitem__,
            (0, 2)
        )

        c = Covariance(c_type=ctypes.c_float)
        _ = c[0, 0]  # Valid access
        nose.tools.assert_raises(
            IndexError,
            c.__getitem__,
            (0, 2)
        )

    def test_set(self):
        m = numpy.ndarray((3, 3))
        # [[ 0 1 2 ]                      [[ 0 2 4 ]
        #  [ 3 4 5 ]  -> should become ->  [ 2 4 6 ]
        #  [ 6 7 8 ]]                      [ 4 6 8 ]]
        m.reshape((9,))[:] = range(9)
        c = Covariance(3, c_type=ctypes.c_double, init_scalar_or_matrix=m)

        # modify some locations
        c[0,1] = 1
        c[2,2] = 3

        nose.tools.assert_equal(c[0,0], 0)
        nose.tools.assert_equal(c[0,1], 1)
        nose.tools.assert_equal(c[0,2], 4)
        nose.tools.assert_equal(c[1,1], 4)
        nose.tools.assert_equal(c[1,2], 6)
        nose.tools.assert_equal(c[2,2], 3)
        nose.tools.assert_equal(c[0,1], c[1,0])
        nose.tools.assert_equal(c[0,2], c[2,0])
        nose.tools.assert_equal(c[1,2], c[2,1])

        # Set in upper triangle and see it reflect in lower
        c[0, 2] = 10.1
        nose.tools.assert_equal(c[2, 0], 10.1)

        # Change something in lower triangle and see it reflected in upper
        c[2, 1] = 20.2
        nose.tools.assert_equal(c[1, 2], 20.2)

        # FLOAT
        c = Covariance(3, c_type=ctypes.c_float, init_scalar_or_matrix=m)

        # modify some locations
        c[0,1] = 1
        c[2,2] = 3

        nose.tools.assert_equal(c[0,0], 0)
        nose.tools.assert_equal(c[0,1], 1)
        nose.tools.assert_equal(c[0,2], 4)
        nose.tools.assert_equal(c[1,1], 4)
        nose.tools.assert_equal(c[1,2], 6)
        nose.tools.assert_equal(c[2,2], 3)
        nose.tools.assert_equal(c[0,1], c[1,0])
        nose.tools.assert_equal(c[0,2], c[2,0])
        nose.tools.assert_equal(c[1,2], c[2,1])

        # Set in upper triangle and see it reflect in lower
        c[0, 2] = 10.1
        nose.tools.assert_almost_equal(c[2, 0], 10.1, 6)

        # Change something in lower triangle and see it reflected in upper
        c[2, 1] = 20.2
        nose.tools.assert_almost_equal(c[1, 2], 20.2, 5)

    def test_set_oob(self):
        # 2x2 covariance mat
        c = Covariance(c_type=ctypes.c_float)
        c[0, 0] = 1  # Valid set
        nose.tools.assert_raises(
            IndexError,
            c.__setitem__,
            (0, 2), 1
        )

        c = Covariance(c_type=ctypes.c_double)
        c[0, 0] = 1  # Valid set
        nose.tools.assert_raises(
            IndexError,
            c.__setitem__,
            (0, 2), 1
        )

    def test_from_cptr(self):
        # Create a new covariance from C function and create new python instance
        # from that pointer
        c_new_func = VitalObject.VITAL_LIB['vital_covariance_3d_new']
        c_new_func.argtypes = [VitalErrorHandle.C_TYPE_PTR]
        c_new_func.restype = Covariance.c_ptr_type(3, ctypes.c_double)
        with VitalErrorHandle() as eh:
            c_ptr = c_new_func(eh)

        c = Covariance(N=3, c_type=ctypes.c_double, from_cptr=c_ptr)
        nose.tools.assert_is(c.C_TYPE_PTR, Covariance.c_ptr_type(3, ctypes.c_double))
        numpy.testing.assert_array_equal(c.to_matrix(), numpy.eye(3))


        c_new_func = VitalObject.VITAL_LIB['vital_covariance_3f_new']
        c_new_func.argtypes = [VitalErrorHandle.C_TYPE_PTR]
        c_new_func.restype = Covariance.c_ptr_type(3, ctypes.c_float)
        with VitalErrorHandle() as eh:
            c_ptr = c_new_func(eh)

        c = Covariance(N=3, c_type=ctypes.c_float, from_cptr=c_ptr)
        nose.tools.assert_is(c.C_TYPE_PTR,Covariance.c_ptr_type(3, ctypes.c_float))
        numpy.testing.assert_array_equal(c.to_matrix(), numpy.eye(3))
