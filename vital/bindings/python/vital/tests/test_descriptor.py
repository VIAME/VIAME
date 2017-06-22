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

vital::descriptor interface tests

"""
import ctypes
import random
import unittest

import nose.tools
import numpy

from vital.exceptions.base import VitalDynamicCastException
from vital.types import Descriptor


class TestDescriptor (unittest.TestCase):

    def test_new(self):
        # Attempt construction using a bunch of random, non-zero integers
        random.seed(0)
        for i in xrange(100):
            n = random.randint(1, 4096)
            Descriptor(n, ctypes.c_double)
            Descriptor(n, ctypes.c_float)

    def test_new_invalid_size(self):
        # Check that we need to pass an integer size.
        nose.tools.assert_raises(
            ctypes.ArgumentError,
            Descriptor, 42.3
        )

    def test_size(self):
        # Check that we can check the size of the descriptor array.
        random.seed(0)
        for i in xrange(100):
            n = random.randint(1, 4096)
            nose.tools.assert_equal(Descriptor(n).size, n)

    def test_num_bytes(self):
        # While not calling the C function, it should still be a correct value
        random.seed(0)
        for i in xrange(100):
            n = random.randint(1, 4096)
            print n,

            nose.tools.assert_equal(
                Descriptor(n, ctypes.c_double).nbytes,
                ctypes.sizeof(ctypes.c_double) * n
            )
            nose.tools.assert_equal(
                Descriptor(n, ctypes.c_float).nbytes,
                ctypes.sizeof(ctypes.c_float) * n
            )

    def test_raw_data(self):
        d = Descriptor(64)
        d[:] = 1
        nose.tools.assert_equal(d.sum(), 64)

        # Check that slicing the array data yields an array with the same
        # values.
        d2 = d[:]
        numpy.testing.assert_almost_equal(d, d2)

        # Check that modifying the sliced array shared the same data as the
        # parent descriptor.
        d2[:] = 2
        numpy.testing.assert_almost_equal(d, d2)

    def test_tobytearray(self):
        # Expect 0-valued descriptor to have 0-valued byte array of the
        # appropriate size
        d = Descriptor(64)
        d[:] = 0
        b = d.tobytearray()
        nose.tools.assert_equal(b.size, d.nbytes)
        nose.tools.assert_equal(b.sum(), 0)
