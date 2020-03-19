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

Tests for Python interface to vital::uid

"""

from kwiver.vital.types import FeatureD, FeatureF, FeatureSet, SimpleFeatureSet
from kwiver.vital.tests.py_helpers import no_call_pure_virtual_method

import nose.tools as nt
import numpy as np


class ConcreteFeatureSet(FeatureSet):
    def __init__(self):
        FeatureSet.__init__(self)

    def size(self):
        return 100

    def features(self):
        return [FeatureD([1, 2], 3, 4, 5)]


class TestVitalFeatureSet(object):
    def test_new(self):
        FeatureSet()

    def test_bad_call_virtual_size(self):
        no_call_pure_virtual_method(FeatureSet().size)

    def test_bad_call_virtual_features(self):
        no_call_pure_virtual_method(FeatureSet().features)

    def test_overriden_size(self):
        nt.assert_equals(ConcreteFeatureSet().size(), 100)

    def test_overriden_features(self):
        nt.assert_equals(ConcreteFeatureSet().features(), [FeatureD([1, 2], 3, 4, 5)])


class TestVitalSimpleFeatureSet(object):
    def _create_features(self):
        return [
            FeatureF(),
            FeatureF([1, 1], 1, 2, 1),
            FeatureD(),
            FeatureD([1, 1], 1, 2, 1),
        ]

    def test_new(self):
        SimpleFeatureSet()
        SimpleFeatureSet(self._create_features())

    def _create_feature_sets(self):
        return (SimpleFeatureSet(), SimpleFeatureSet(self._create_features()))

    def test_size(self):
        empty, nonempty = self._create_feature_sets()

        nt.assert_equals(empty.size(), 0)
        nt.assert_equals(nonempty.size(), 4)

    def test_features(self):
        empty, nonempty = self._create_feature_sets()

        np.testing.assert_array_equal(empty.features(), [])
        np.testing.assert_array_equal(nonempty.features(), self._create_features())

        # Test that elements can be modified using the features method, and are not copied
        f = nonempty.features()[0]
        f.scale += 1
        nt.ok_(nonempty.features()[0] == f)
