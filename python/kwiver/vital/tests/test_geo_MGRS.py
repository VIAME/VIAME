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
SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

==============================================================================

Tests for geo_MGRS interface

"""

import unittest
from kwiver.vital.types import GeoMGRS


class TestVitalGeoMGRS(unittest.TestCase):
    def _create_geo_mgrs(self):
        return (GeoMGRS(), GeoMGRS(""), GeoMGRS("foo"), GeoMGRS("1.234"))

    def test_new(self):
        GeoMGRS()
        GeoMGRS("")
        GeoMGRS("foo")
        GeoMGRS("1.234")

    def test_initial_valid_and_empty(self):
        empty1, empty2, foo, num = self._create_geo_mgrs()

        self.assertFalse(empty1.is_valid())
        self.assertFalse(empty2.is_valid())
        self.assertTrue(foo.is_valid())
        self.assertTrue(num.is_valid())

        self.assertTrue(empty1.is_empty())
        self.assertTrue(empty2.is_empty())
        self.assertFalse(foo.is_empty())
        self.assertFalse(num.is_empty())

    def test_initial_coord(self):
        empty1, empty2, foo, num = self._create_geo_mgrs()

        self.assertEqual(empty1.coord(), "")
        self.assertEqual(empty2.coord(), "")
        self.assertEqual(foo.coord(), "foo")
        self.assertEqual(num.coord(), "1.234")

    def test_set_get_coord(self):
        for gm in list(self._create_geo_mgrs()):
            gm_cpy = gm.set_coord("test_str12345")
            self.assertEqual(gm.coord(), "test_str12345")
            self.assertEqual(gm_cpy.coord(), "test_str12345")
            self.assertFalse(gm.is_empty())
            self.assertTrue(gm.is_valid())
            self.assertFalse(gm_cpy.is_empty())
            self.assertTrue(gm_cpy.is_valid())

            gm_cpy = gm.set_coord("another_test_str")
            self.assertEqual(gm.coord(), "another_test_str")
            self.assertEqual(gm_cpy.coord(), "another_test_str")
            self.assertFalse(gm.is_empty())
            self.assertTrue(gm.is_valid())
            self.assertFalse(gm_cpy.is_empty())
            self.assertTrue(gm_cpy.is_valid())

            gm_cpy = gm.set_coord("")
            self.assertEqual(gm.coord(), "")
            self.assertEqual(gm_cpy.coord(), "")
            self.assertTrue(gm.is_empty())
            self.assertFalse(gm.is_valid())
            self.assertTrue(gm_cpy.is_empty())
            self.assertFalse(gm_cpy.is_valid())

    def test_equals(self):
        gm1, gm2 = GeoMGRS(), GeoMGRS()
        self.assertTrue(gm1 == gm2)

        # Check copies are equal
        gm1_cpy = gm1.set_coord("test_str12345")
        self.assertFalse(gm1 == gm2)
        self.assertTrue(gm1_cpy == gm1)
        self.assertFalse(gm1_cpy == gm2)

        gm2.set_coord("test_str12345")
        self.assertTrue(gm1 == gm2)
        self.assertTrue(gm1_cpy == gm1)
        self.assertTrue(gm1_cpy == gm2)

    def test_not_equals(self):
        gm1, gm2 = GeoMGRS(), GeoMGRS()
        self.assertFalse(gm1 != gm2)

        gm1_cpy = gm1.set_coord("test_str12345")
        self.assertTrue(gm1 != gm2)
        self.assertFalse(gm1_cpy != gm1)
        self.assertTrue(gm1_cpy != gm2)

        gm2.set_coord("test_str12345")
        self.assertFalse(gm1 != gm2)
        self.assertFalse(gm1_cpy != gm1)
        self.assertFalse(gm1_cpy != gm2)

    def test_to_str_empty(self):
        gm = GeoMGRS()
        self.assertEqual(str(gm), "[MGRS: ]")

    def test_to_str(self):
        gm = GeoMGRS("test_coord_1.234")
        self.assertEqual(str(gm), "[MGRS: test_coord_1.234]")
