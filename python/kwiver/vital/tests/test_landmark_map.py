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

Tests for Landmark_Map interface

"""

import numpy as np
import numpy.testing as npt
import nose.tools as nt
import unittest

from kwiver.vital.types import (
    LandmarkF,
    LandmarkD,
    LandmarkMap,
    SimpleLandmarkMap,
)

class TestSimpleLandmarkMap(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.landmark1 = LandmarkF([0, 1, 2])
        self.landmark2 = LandmarkF([0, 1, 3])
        self.landmarks_dict = {0:self.landmark1, 1:self.landmark2}

    def test_inherits(self):
        nt.ok_(issubclass(SimpleLandmarkMap, LandmarkMap))

    def test_construct(self):
        SimpleLandmarkMap()
        SimpleLandmarkMap(self.landmarks_dict)

    def test_methods(self):
        sm = SimpleLandmarkMap(self.landmarks_dict)
        nt.assert_equal(sm.size(), 2)
        nt.assert_dict_equal(sm.landmarks(), self.landmarks_dict)
