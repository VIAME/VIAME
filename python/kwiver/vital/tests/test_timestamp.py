"""
ckwg +31
Copyright 2018-2020 by Kitware, Inc.
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
from kwiver.vital.types import Timestamp


class TestTimestamp(unittest.TestCase):
    def test_new(self):
        Timestamp()
        Timestamp(1234000000, 1)

    def test_time(self):
        t1 = Timestamp()
        t1.set_time_seconds(1234)
        self.assertEqual(t1.get_time_seconds(), 1234)
        self.assertEqual(t1.get_time_usec(), 1234000000)

        t1.set_time_usec(4321000000)
        self.assertEqual(t1.get_time_seconds(), 4321)
        self.assertEqual(t1.get_time_usec(), 4321000000)

        t2 = Timestamp(1234000000, 1)
        self.assertEqual(t2.get_time_seconds(), 1234)
        self.assertEqual(t2.get_time_usec(), 1234000000)

    def test_frame(self):
        t1 = Timestamp()
        t1.set_frame(1)
        self.assertEqual(t1.get_frame(), 1)

        t2 = Timestamp(1234000000, 1)
        self.assertEqual(t2.get_frame(), 1)

    def test_valid(self):
        t1 = Timestamp()
        self.assertFalse(t1.is_valid())
        self.assertFalse(t1.has_valid_time())
        self.assertFalse(t1.has_valid_frame())

        t1.set_time_seconds(1234)
        self.assertFalse(t1.is_valid())
        t1.set_frame(1)
        self.assertTrue(t1.is_valid())
        self.assertTrue(t1.has_valid_time())
        self.assertTrue(t1.has_valid_frame())

        t1.set_invalid()
        self.assertFalse(t1.is_valid())
        self.assertFalse(t1.has_valid_time())
        self.assertFalse(t1.has_valid_frame())

    def test_operators(self):
        t1 = Timestamp(100, 1)
        t2 = Timestamp(100, 1)
        t3 = Timestamp(200, 2)

        self.assertTrue(t1 == t2)
        self.assertTrue(t1 != t3)
        self.assertTrue(t1 < t3)
        self.assertTrue(t3 > t1)
        self.assertTrue(t1 <= t2)
        self.assertTrue(t1 <= t3)
        self.assertTrue(t1 >= t2)
        self.assertTrue(t3 >= t1)

        self.assertFalse(t1 == t3)
        self.assertFalse(t1 != t2)
        self.assertFalse(t3 < t1)
        self.assertFalse(t1 < t2)
        self.assertFalse(t1 > t3)
        self.assertFalse(t1 > t2)
        self.assertFalse(t3 <= t1)
        self.assertFalse(t1 >= t3)
