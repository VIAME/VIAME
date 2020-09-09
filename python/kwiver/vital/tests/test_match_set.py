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

Tests for the vital class match_set

"""

import nose.tools as nt
import numpy.testing as npt
import numpy as np
import unittest

from kwiver.vital.tests.cpp_helpers import match_set_helpers as msh

from kwiver.vital.types import (
    MatchVector,
    BaseMatchSet,
    MatchSet,
)


class TestSimpleMatchSet(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.match_a = (3, 1)
        self.match_b = (4, 1)
        self.match_c = (5, 9)
        self.match_set = [self.match_a, self.match_b, self.match_c]

    def test_constructor(self):
        MatchSet()
        MatchSet(self.match_set)

    def test_size(self):
        m = MatchSet(self.match_set)
        nt.assert_equal(m.size(), 3)

    def test_matches(self):
        m = MatchSet(self.match_set)
        npt.assert_array_equal(m.matches(), self.match_set)

    def test_py_helpers(self):
        m = MatchSet(self.match_set)
        nt.assert_equal(str(m), "<MatchSet>")
        nt.assert_equal(repr(m)[1:9], "MatchSet")

class TestBaseMatchSet(unittest.TestCase):
    @classmethod
    def setUp(self):
        pass

    def test_constructor(self):
        BaseMatchSet()

class MatchSetInherit(BaseMatchSet):
    def __init__(self):
        BaseMatchSet.__init__(self)
        self.match_a = (3, 1)
        self.match_b = (4, 1)
        self.match_c = (5, 9)
        self.match_set = [self.match_a, self.match_b, self.match_c]
    def size(self):
        return 299
    def matches(self):
        return self.match_set
    def __str__(self):
        return "{} {} {}".format(self.match_a, self.match_b, self.match_c)
    def __repr__(self):
        return "{} {} {}".format(self.match_a, self.match_b, self.match_c)

class TestMatchSetInherit(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.match_a = (3, 1)
        self.match_b = (4, 1)
        self.match_c = (5, 9)
        self.match_set = [self.match_a, self.match_b, self.match_c]

    def test_init(self):
        MatchSetInherit()

    def test_inherited_virts(self):
        b = MatchSetInherit()
        nt.assert_equal(msh.call_size(b), 299)
        npt.assert_array_equal(msh.call_matches(b), self.match_set)
