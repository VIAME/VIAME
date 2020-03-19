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

Tests for Python interface to vital::transform_2d

"""
import numpy as np

import nose.tools as nt

from kwiver.vital.types import Transform2D

class SimpleTransform2D(Transform2D):
    def __init__(self, arr):
        Transform2D.__init__(self)
        self.arr = arr

    def clone(self):
        return SimpleTransform2D(self.arr.copy())

    def map(self, p):
        return p + 5

    def inverse_(self):
        return SimpleTransform2D(1/self.arr)

class TestVitalTransform2D(object):
    # Note that clone and inverse_ are skipped. See binding code for explanation

    def test_bad_call_virtual_map(self):
        t = Transform2D()
        with nt.assert_raises_regexp(
            RuntimeError,
            "Tried to call pure virtual function",
        ):
            t.map(np.array([2, 4]))

    def test_is_instance(self):
        st = SimpleTransform2D(np.array([2, 4]))
        nt.ok_(isinstance(st, Transform2D))

class TestVitalTransform2DSubclass(object):
    def test_inverse_(self):
        st = SimpleTransform2D(np.array([2, 4]))
        st_inverse = st.inverse()
        np.testing.assert_array_equal(st_inverse.arr, np.array([0.5, 0.25]))
        # Make sure instance wasn't sliced
        nt.ok_(isinstance(st_inverse, SimpleTransform2D))

    def test_clone(self):
        st = SimpleTransform2D(np.array([2, 4]))
        st_clone = st.clone()

    def test_map(self):
        st = SimpleTransform2D(np.array([2, 4]))
        np.testing.assert_array_equal(st.map(np.array([-5, 5])), np.array([0, 10]))
