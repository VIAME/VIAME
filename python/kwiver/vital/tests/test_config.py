#!/usr/bin/env python
# ckwg +28
# Copyright 2019-2020 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from kwiver.vital.config import Config, ConfigKeys, empty_config
from kwiver.vital.tests.py_helpers import create_geo_poly
from kwiver.vital.types import GeoPolygon

import numpy as np
import unittest


class TestVitalConfig(unittest.TestCase):
    def test_create(self):
        self.assertTrue(len(empty_config()) == 0)
        ConfigKeys()

    def test_api_calls(self):
        Config.block_sep()
        Config.global_value

    def test_has_value(self):
        c = empty_config()
        keya = "keya"
        keyb = "keyb"
        valuea = "valuea"
        c.set_value(keya, valuea)
        self.assertTrue(c.has_value(keya))
        self.assertTrue(not c.has_value(keyb))

    def test_get_value(self):
        c = empty_config()
        keya = "keya"
        valuea = "valuea"
        c.set_value(keya, valuea)
        get_valuea = c.get_value(keya)
        self.assertEqual(valuea, get_valuea)

    def test_get_value_nested(self):
        c = empty_config()
        keya = "keya"
        keyb = "keyb"
        valuea = "valuea"
        c.set_value(keya + Config.block_sep() + keyb, valuea)
        nc = c.subblock(keya)
        get_valuea = nc.get_value(keyb)
        self.assertEqual(valuea, get_valuea)

    def test_get_value_no_exist(self):
        with self.assertRaises(RuntimeError):
            c = empty_config()
            keya = "keya"
            c.get_value(keya)

    def test_default_value(self):
        c = empty_config()
        keyb = "keyb"
        valueb = "valueb"
        get_valueb = c.get_value(keyb, valueb)
        self.assertEqual(valueb, get_valueb)

    def test_unset_value(self):
        with self.assertRaises(RuntimeError):
            c = empty_config()
            keya = "keya"
            valuea = "valuea"
            c.set_value(keya, valuea)
            c.unset_value(keya)
            c.get_value(keya)

    def test_available_values(self):
        c = empty_config()
        keya = "keya"
        keyb = "keyb"

        valuea = "valuea"
        valueb = "valueb"

        c.set_value(keya, valuea)
        c.set_value(keyb, valueb)

        avail = c.available_values()
        self.assertEqual(len(avail), 2)

    def test_available_values_are_iterable(self):
        c = empty_config()
        self.assertTrue(hasattr(c, "__getitem__"))

    def test_read_only(self):
        with self.assertRaises(RuntimeError):
            c = empty_config()
            keya = "keya"
            valuea = "valuea"
            valueb = "valueb"
            c.set_value(keya, valuea)
            c.mark_read_only(keya)
            c.set_value(keya, valueb)

    def test_read_only_unset(self):
        with self.assertRaises(RuntimeError):
            c = empty_config()
            keya = "keya"
            valuea = "valuea"
            c.set_value(keya, valuea)
            c.mark_read_only(keya)
            c.unset_value(keya)

    def test_subblock(self):
        c = empty_config()
        block1 = "block1"
        block2 = "block2"

        keya = "keya"
        keyb = "keyb"
        keyc = "keyc"

        valuea = "valuea"
        valueb = "valueb"
        valuec = "valuec"

        c.set_value(block1 + Config.block_sep() + keya, valuea)
        c.set_value(block1 + Config.block_sep() + keyb, valueb)
        c.set_value(block2 + Config.block_sep() + keyc, valuec)

        d = c.subblock(block1)
        get_valuea = d.get_value(keya)
        get_valueb = d.get_value(keyb)

        self.assertEqual(valuea, get_valuea, "Subblock does not inherit expected keys")
        self.assertEqual(valueb, get_valueb, "Subblock does not inherit expected keys")
        self.assertEqual(d.has_value(keyc), False, "Subblock inherited unrelated key")

    def test_subblock_view(self):
        c = empty_config()

        block1 = "block1"
        block2 = "block2"

        keya = "keya"
        keyb = "keyb"

        valuea = "valuea"
        valueb = "valueb"

        c.set_value(block1 + Config.block_sep() + keya, valuea)
        c.set_value(block2 + Config.block_sep() + keyb, valueb)

        d = c.subblock_view(block1)

        self.assertTrue(d.has_value(keya), "Subblock does not inherit expected keys")

        self.assertEqual(d.has_value(keyb), False, "Subblock inherited unrelated key")

        c.set_value(block1 + Config.block_sep() + keya, valueb)

        get_valuea1 = d.get_value(keya)

        self.assertEqual(valueb, get_valuea1, "Subblock view persisted a changed value")

        d.set_value(keya, valuea)

        get_valuea2 = d.get_value(keya)
        self.assertEqual(
            valuea, get_valuea2, "Subblock view set value was not changed in parent"
        )

    def test_merge_config(self):
        c = empty_config()
        d = empty_config()
        keya = "keya"
        keyb = "keyb"
        keyc = "keyc"
        valuea = "valuea"
        valueb = "valueb"
        valuec = "valuec"
        c.set_value(keya, valuea)
        c.set_value(keyb, valuea)
        d.set_value(keyb, valueb)
        d.set_value(keyc, valuec)
        c.merge_config(d)
        get_valuea = c.get_value(keya)
        get_valueb = c.get_value(keyb)
        get_valuec = c.get_value(keyc)
        self.assertEqual(valuea, get_valuea, "Unmerged key changed")
        self.assertEqual(valueb, get_valueb, "Conflicting key was not overwritten")
        self.assertEqual(valuec, get_valuec, "New key did not appear")

    def test_getitem(self):
        c = empty_config()
        key = "key"
        value = "oldvalue"
        c[key] = value

        self.assertEqual(c[key], value)
        self.assertTrue(key in c, "{0} is not in config after insertion".format(key))

        value = "newvalue"
        origvalue = "newvalue"
        c[key] = value
        value = "replacedvalue"

        self.assertEqual(c[key], origvalue, "Value was overwritten")

    def test_invalid_getitem(self):
        with self.assertRaises(KeyError):
            c = empty_config()
            key = "key"
            value = c[key]

    def test_delitem(self):
        c = empty_config()
        key = "key"
        value = "oldvalue"
        c[key] = value
        del c[key]
        self.assertEqual(c.has_value(key), False, "The key was not deleted")

    def test_invalid_delitem(self):
        with self.assertRaises(KeyError):
            c = empty_config()
            key = "key"
            del c[key]

    def test_implicit_string_conversion_of_values(self):
        c = empty_config()
        key = "key"
        value = 10
        c[key] = value
        self.assertEqual(c[key], str(value))

    ##################geo_poly tests###################

    def check_pts_equal(self, gp1, gp2):
        pts_in = gp1.polygon().get_vertices()
        pts_out = gp2.polygon().get_vertices()

        self.assertEqual(len(pts_in), len(pts_out))
        for loc_in, loc_out in zip(pts_in, pts_out):
            np.testing.assert_array_almost_equal(loc_in, loc_out, decimal=15)

    def test_get_value_geo_poly(self):
        c = empty_config()
        keya = "keya"
        valuea = create_geo_poly()

        c.set_value_geo_poly(keya, valuea)
        get_valuea = c.get_value_geo_poly(keya)
        self.check_pts_equal(valuea, get_valuea)

    def test_get_value_empty_geo_poly(self):
        c = empty_config()
        keya = "keya"
        valuea = GeoPolygon()

        c.set_value_geo_poly(keya, valuea)
        get_valuea = c.get_value_geo_poly(keya)

        self.assertTrue(valuea.is_empty())
        self.assertTrue(get_valuea.is_empty())

    def test_default_value_geo_poly(self):
        c = empty_config()
        keyb = "keyb"
        valueb = create_geo_poly()

        get_valueb = c.get_value_geo_poly(keyb, valueb)

        self.check_pts_equal(valueb, get_valueb)

    def test_get_value_geo_poly_no_exist(self):
        with self.assertRaises(RuntimeError):
            c = empty_config()
            keya = "keya"
            c.get_value_geo_poly(keya)

    def test_get_value_geo_poly_nested(self):
        c = empty_config()
        keya = "keya"
        keyb = "keyb"
        valuea = create_geo_poly()
        c.set_value_geo_poly(keya + Config.block_sep() + keyb, valuea)
        nc = c.subblock(keya)
        get_valuea = nc.get_value_geo_poly(keyb)

        self.check_pts_equal(valuea, get_valuea)

    def test_unset_geo_poly(self):
        with self.assertRaises(RuntimeError):
            c = empty_config()
            keya = "keya"
            valuea = create_geo_poly()
            c.set_value_geo_poly(keya, valuea)
            c.unset_value(keya)
            c.get_value_geo_poly(keya)

    def test_set_get_value_wrong_type(self):
        c = empty_config()
        key = "key"
        value_gp = create_geo_poly()
        value_str = "value_str"

        with self.assertRaises(TypeError):
            c.set_value_geo_poly(key, value_str)
        with self.assertRaises(TypeError):
            c.set_value(key, value_gp)

        c.set_value(key, value_str)

        with self.assertRaises(RuntimeError):
            c.get_value_geo_poly(key)

    def test_get_geo_poly_str(self):
        c = empty_config()
        keya = "key"
        valuea = create_geo_poly()

        c.set_value_geo_poly(keya, valuea)
        str_out = c.get_value(keya)
        self.assertTrue(isinstance(str_out, str))


################end geo_poly tests#################
