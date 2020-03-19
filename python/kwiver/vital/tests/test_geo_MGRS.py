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

import nose.tools as nt

from kwiver.vital.types import GeoMGRS


class TestVitalGeoMGRS(object):
    def _create_geo_mgrs(self):
        return (GeoMGRS(), GeoMGRS(""), GeoMGRS("foo"), GeoMGRS("1.234"))

    def test_new(self):
        GeoMGRS()
        GeoMGRS("")
        GeoMGRS("foo")
        GeoMGRS("1.234")

    def test_initial_valid_and_empty(self):
        empty1, empty2, foo, num = self._create_geo_mgrs()

        nt.assert_false(empty1.is_valid())
        nt.assert_false(empty2.is_valid())
        nt.assert_true( foo.is_valid())
        nt.assert_true( num.is_valid())

        nt.assert_true( empty1.is_empty())
        nt.assert_true( empty2.is_empty())
        nt.assert_false(foo.is_empty())
        nt.assert_false(num.is_empty())

    def test_initial_coord(self):
        empty1, empty2, foo, num = self._create_geo_mgrs()

        nt.assert_equal(empty1.coord(), "")
        nt.assert_equal(empty2.coord(), "")
        nt.assert_equal(foo.coord(), "foo")
        nt.assert_equal(num.coord(), "1.234")

    def test_set_get_coord(self):
        for gm in list(self._create_geo_mgrs()):
            gm_cpy = gm.set_coord("test_str12345")
            nt.assert_equal(gm.coord(), "test_str12345")
            nt.assert_equal(gm_cpy.coord(), "test_str12345")
            nt.assert_false(gm.is_empty())
            nt.assert_true( gm.is_valid())
            nt.assert_false(gm_cpy.is_empty())
            nt.assert_true( gm_cpy.is_valid())

            gm_cpy = gm.set_coord("another_test_str")
            nt.assert_equal(gm.coord(), "another_test_str")
            nt.assert_equal(gm_cpy.coord(), "another_test_str")
            nt.assert_false(gm.is_empty())
            nt.assert_true( gm.is_valid())
            nt.assert_false(gm_cpy.is_empty())
            nt.assert_true( gm_cpy.is_valid())

            gm_cpy = gm.set_coord("")
            nt.assert_equal(gm.coord(), "")
            nt.assert_equal(gm_cpy.coord(), "")
            nt.assert_true( gm.is_empty())
            nt.assert_false(gm.is_valid())
            nt.assert_true( gm_cpy.is_empty())
            nt.assert_false(gm_cpy.is_valid())

    def test_equals(self):
        gm1, gm2 = GeoMGRS(), GeoMGRS()
        nt.assert_true(gm1 == gm2)

        # Check copies are equal
        gm1_cpy = gm1.set_coord("test_str12345")
        nt.assert_false(gm1 == gm2)
        nt.assert_true( gm1_cpy == gm1)
        nt.assert_false(gm1_cpy == gm2)

        gm2.set_coord("test_str12345")
        nt.assert_true(gm1 == gm2)
        nt.assert_true(gm1_cpy == gm1)
        nt.assert_true(gm1_cpy == gm2)

    def test_not_equals(self):
        gm1, gm2 = GeoMGRS(), GeoMGRS()
        nt.assert_false(gm1 != gm2)

        gm1_cpy = gm1.set_coord("test_str12345")
        nt.assert_true( gm1 != gm2)
        nt.assert_false(gm1_cpy != gm1)
        nt.assert_true( gm1_cpy != gm2)

        gm2.set_coord("test_str12345")
        nt.assert_false(gm1 != gm2)
        nt.assert_false(gm1_cpy != gm1)
        nt.assert_false(gm1_cpy != gm2)

    def test_to_str_empty(self):
        gm = GeoMGRS()
        nt.assert_equal(str(gm), "[MGRS: ]")

    def test_to_str(self):
        gm = GeoMGRS("test_coord_1.234")
        nt.assert_equal(str(gm), "[MGRS: test_coord_1.234]")
