#!/usr/bin/env python
#ckwg +28
# Copyright 2019 by Kitware, Inc.
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
from kwiver.vital.config import config
import nose.tools

class TestVitalConfig(object):
    def test_create(self):
        nose.tools.ok_(len(config.empty_config()) == 0)
        config.ConfigKeys()

    def test_api_calls(self):
        config.Config.block_sep()
        config.Config.global_value

    def test_has_value(self):
        c = config.empty_config()
        keya = 'keya'
        keyb = 'keyb'
        valuea = 'valuea'
        c.set_value(keya, valuea)
        nose.tools.ok_(c.has_value(keya))
        nose.tools.ok_(not c.has_value(keyb))

    def test_get_value(self):
        c = config.empty_config()
        keya = 'keya'
        valuea = 'valuea'
        c.set_value(keya, valuea)
        get_valuea = c.get_value(keya)
        nose.tools.assert_equal(valuea, get_valuea)

    def test_get_value_nested(self):
        c = config.empty_config()
        keya = 'keya'
        keyb = 'keyb'
        valuea = 'valuea'
        c.set_value(keya + config.Config.block_sep() + keyb, valuea)
        nc = c.subblock(keya)
        get_valuea = nc.get_value(keyb)
        nose.tools.assert_equal(valuea, get_valuea)

    @nose.tools.raises(RuntimeError)
    def test_get_value_no_exist(self):
        c = config.empty_config()
        keya = 'keya'
        c.get_value(keya)

    def test_default_value(self):
        c = config.empty_config()
        keya = 'keya'
        keyb = 'keyb'
        valueb = 'valueb'
        get_valueb = c.get_value(keyb, valueb)
        nose.tools.assert_equal(valueb, get_valueb)

    @nose.tools.raises(RuntimeError)
    def test_unset_value(self):
        c = config.empty_config()
        keya = 'keya'
        valuea = 'valuea'
        c.set_value(keya, valuea)
        c.unset_value(keya)
        c.get_value(keya)

    def test_available_values(self):
        c = config.empty_config()
        keya = 'keya'
        keyb = 'keyb'

        valuea = 'valuea'
        valueb = 'valueb'

        c.set_value(keya, valuea)
        c.set_value(keyb, valueb)

        avail = c.available_values()
        nose.tools.assert_equal(len(avail), 2)

    def test_available_values_are_iterable(self):
        c = config.empty_config()
        nose.tools.ok_(hasattr(c, "__getitem__"))

    @nose.tools.raises(RuntimeError)
    def test_read_only(self):
        c = config.empty_config()
        keya = 'keya'
        valuea = 'valuea'
        valueb = 'valueb'
        c.set_value(keya, valuea)
        c.mark_read_only(keya)
        c.set_value(keya, valueb)

    @nose.tools.raises(RuntimeError)
    def test_read_only_unset(self):
        c = config.empty_config()
        keya = 'keya'
        valuea = 'valuea'
        c.set_value(keya, valuea)
        c.mark_read_only(keya)
        c.unset_value(keya)

    def test_subblock(self):
        c = config.empty_config()
        block1 = 'block1'
        block2 = 'block2'

        keya = 'keya'
        keyb = 'keyb'
        keyc = 'keyc'

        valuea = 'valuea'
        valueb = 'valueb'
        valuec = 'valuec'

        c.set_value(block1 + config.Config.block_sep() + keya, valuea)
        c.set_value(block1 + config.Config.block_sep() + keyb, valueb)
        c.set_value(block2 + config.Config.block_sep() + keyc, valuec)

        d = c.subblock(block1)
        get_valuea = d.get_value(keya)
        get_valueb = d.get_value(keyb)

        nose.tools.assert_equal(valuea, get_valuea,
                                "Subblock does not inherit expected keys")
        nose.tools.assert_equal(valueb, get_valueb,
                                "Subblock does not inherit expected keys")
        nose.tools.assert_equal(d.has_value(keyc), False,
                                "Subblock inherited unrelated key" )

    def test_subblock_view(self):
        c = config.empty_config()

        block1 = 'block1'
        block2 = 'block2'

        keya = 'keya'
        keyb = 'keyb'

        valuea = 'valuea'
        valueb = 'valueb'

        c.set_value(block1 + config.Config.block_sep() + keya, valuea)
        c.set_value(block2 + config.Config.block_sep() + keyb, valueb)

        d = c.subblock_view(block1)

        nose.tools.ok_(d.has_value(keya),
                       "Subblock does not inherit expected keys")

        nose.tools.assert_equal(d.has_value(keyb), False,
                               "Subblock inherited unrelated key")

        c.set_value(block1 + config.Config.block_sep() + keya, valueb)

        get_valuea1 = d.get_value(keya)

        nose.tools.assert_equal(valueb, get_valuea1,
                               "Subblock view persisted a changed value")

        d.set_value(keya, valuea)

        get_valuea2 = d.get_value(keya)
        nose.tools.assert_equal(valuea, get_valuea2,
                               "Subblock view set value was not changed in parent")

    def test_merge_config(self):
        c = config.empty_config()
        d = config.empty_config()
        keya = 'keya'
        keyb = 'keyb'
        keyc = 'keyc'
        valuea = 'valuea'
        valueb = 'valueb'
        valuec = 'valuec'
        c.set_value(keya, valuea)
        c.set_value(keyb, valuea)
        d.set_value(keyb, valueb)
        d.set_value(keyc, valuec)
        c.merge_config(d)
        get_valuea = c.get_value(keya)
        get_valueb = c.get_value(keyb)
        get_valuec = c.get_value(keyc)
        nose.tools.assert_equal(valuea, get_valuea, "Unmerged key changed")
        nose.tools.assert_equal(valueb, get_valueb, "Conflicting key was not overwritten")
        nose.tools.assert_equal(valuec, get_valuec, "New key did not appear")


    def test_getitem(self):
        c = config.empty_config()
        key = 'key'
        value = 'oldvalue'
        c[key] = value

        nose.tools.assert_equal(c[key], value)
        nose.tools.ok_(key in c, "{0} is not in config after insertion".format(key))

        value = 'newvalue'
        origvalue = 'newvalue'
        c[key] = value
        value = 'replacedvalue'

        nose.tools.assert_equal(c[key], origvalue, "Value was overwritten")

    @nose.tools.raises(KeyError)
    def test_invalid_getitem(self):
        c = config.empty_config()
        key = 'key'
        value = c[key]

    def test_delitem(self):
        c = config.empty_config()
        key = 'key'
        value = 'oldvalue'
        c[key] = value
        del c[key]
        nose.tools.assert_equal(c.has_value(key), False, "The key was not deleted")

    @nose.tools.raises(KeyError)
    def test_invalid_delitem(self):
        c = config.empty_config()
        key = 'key'
        del c[key]

    def test_implicit_string_conversion_of_values(self):
        c = config.empty_config()
        key = 'key'
        value = 10
        c[key] = value
        nose.tools.assert_equal(c[key], str(value))
