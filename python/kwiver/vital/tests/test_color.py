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

Tests for RGBColor interface

"""

import unittest

import numpy

from kwiver.vital.types import RGBColor


class TestRGBColor(unittest.TestCase):
    def test_new(self):
        RGBColor()
        RGBColor(1, 2, 3)
        # Should integer-cast
        RGBColor(10, 20.5, 30.3)
        # Will overflow, but otherwise valid
        RGBColor(400, 400, 400)

    def test_r(self):
        self.assertEqual(RGBColor().r, 255)
        self.assertEqual(RGBColor(r=0).r, 0)
        self.assertEqual(RGBColor(r=44.4).r, 44)
        self.assertEqual(RGBColor(r=400).r, 400 - 256)

    def test_g(self):
        self.assertEqual(RGBColor().g, 255)
        self.assertEqual(RGBColor(g=0).g, 0)
        self.assertEqual(RGBColor(g=44.4).g, 44)
        self.assertEqual(RGBColor(g=400).g, 400 - 256)

    def test_b(self):
        self.assertEqual(RGBColor().b, 255)
        self.assertEqual(RGBColor(b=0).b, 0)
        self.assertEqual(RGBColor(b=44.4).b, 44)
        self.assertEqual(RGBColor(b=400).b, 400 - 256)

    def test_getitem_access(self):
        c = RGBColor(10, 20, 30)
        self.assertTrue(c.r == c[0] == 10)
        self.assertTrue(c.g == c[1] == 20)
        self.assertTrue(c.b == c[2] == 30)

    def test_getitem_access_IndexError(self):
        c = RGBColor(10, 20, 30)
        with self.assertRaises(IndexError):
            c[4]

    def test_equality(self):
        c1 = RGBColor()
        c2 = RGBColor()
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)

    def test_repr(self):
        c = RGBColor()
        self.assertEqual(repr(c), "RGBColor{255, 255, 255}")
