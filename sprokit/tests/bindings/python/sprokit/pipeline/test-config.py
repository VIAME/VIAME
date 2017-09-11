#!/usr/bin/env python
#ckwg +28
# Copyright 2011-2013 by Kitware, Inc.
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


def test_import():
    try:
        import sprokit.pipeline.config
    except:
        test_error("Failed to import the config module")


def test_create():
    from sprokit.pipeline import config

    try:
        config.empty_config()
    except:
        test_error("Failed to create an empty configuration")

    config.ConfigKey()
    config.ConfigKeys()
    config.ConfigDescription()
    config.ConfigValue()


def test_api_calls():
    from sprokit.pipeline import config

    config.Config.block_sep
    config.Config.global_value


def test_has_value():
    from sprokit.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valuea = 'valuea'

    c.set_value(keya, valuea)

    if not c.has_value(keya):
        test_error("Block does not have value which was set")

    if c.has_value(keyb):
        test_error("Block has value which was not set")


def test_get_value():
    from sprokit.pipeline import config

    c = config.empty_config()

    keya = 'keya'

    valuea = 'valuea'

    c.set_value(keya, valuea)

    get_valuea = c.get_value(keya)

    if not valuea == get_valuea:
        test_error("Did not retrieve value that was set")


def test_get_value_nested():
    from sprokit.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valuea = 'valuea'

    c.set_value(keya + config.Config.block_sep + keyb, valuea)

    nc = c.subblock(keya)

    get_valuea = nc.get_value(keyb)

    if not valuea == get_valuea:
        test_error("Did not retrieve value that was set")


def test_get_value_no_exist():
    from sprokit.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valueb = 'valueb'

    expect_exception('retrieving an unset value', BaseException,
                     c.get_value, keya)

    get_valueb = c.get_value(keyb, valueb)

    if not valueb == get_valueb:
        test_error("Did not retrieve default when requesting unset value")


def test_unset_value():
    from sprokit.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valuea = 'valuea'
    valueb = 'valueb'

    c.set_value(keya, valuea)
    c.set_value(keyb, valueb)

    c.unset_value(keya)

    expect_exception('retrieving an unset value', BaseException,
                     c.get_value, keya)

    get_valueb = c.get_value(keyb)

    if not valueb == get_valueb:
        test_error("Did not retrieve value when requesting after an unrelated unset")


def test_available_values():
    from sprokit.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valuea = 'valuea'
    valueb = 'valueb'

    c.set_value(keya, valuea)
    c.set_value(keyb, valueb)

    avail = c.available_values()

    if not len(avail) == 2:
        test_error("Did not retrieve correct number of keys")

    try:
        for val in avail:
            pass
    except:
        test_error("Available values is not iterable")


def test_read_only():
    from sprokit.pipeline import config

    c = config.empty_config()

    keya = 'keya'

    valuea = 'valuea'
    valueb = 'valueb'

    c.set_value(keya, valuea)

    c.mark_read_only(keya)

    expect_exception('setting a read only value', BaseException,
                     c.set_value, keya, valueb)

    get_valuea = c.get_value(keya)

    if not valuea == get_valuea:
        test_error("Read only value changed")


def test_read_only_unset():
    from sprokit.pipeline import config

    c = config.empty_config()

    keya = 'keya'

    valuea = 'valuea'

    c.set_value(keya, valuea)

    c.mark_read_only(keya)

    expect_exception('unsetting a read only value', BaseException,
                     c.unset_value, keya)

    get_valuea = c.get_value(keya)

    if not valuea == get_valuea:
        test_error("Read only value was unset")


def test_subblock():
    from sprokit.pipeline import config

    c = config.empty_config()

    block1 = 'block1'
    block2 = 'block2'

    keya = 'keya'
    keyb = 'keyb'
    keyc = 'keyc'

    valuea = 'valuea'
    valueb = 'valueb'
    valuec = 'valuec'

    c.set_value(block1 + config.Config.block_sep + keya, valuea)
    c.set_value(block1 + config.Config.block_sep + keyb, valueb)
    c.set_value(block2 + config.Config.block_sep + keyc, valuec)

    d = c.subblock(block1)

    get_valuea = d.get_value(keya)

    if not valuea == get_valuea:
        test_error("Subblock does not inherit expected keys")

    get_valueb = d.get_value(keyb)

    if not valueb == get_valueb:
        test_error("Subblock does not inherit expected keys")

    if d.has_value(keyc):
        test_error("Subblock inherited unrelated key")


def test_subblock_view():
    from sprokit.pipeline import config

    c = config.empty_config()

    block1 = 'block1'
    block2 = 'block2'

    keya = 'keya'
    keyb = 'keyb'
    keyc = 'keyc'

    valuea = 'valuea'
    valueb = 'valueb'
    valuec = 'valuec'

    c.set_value(block1 + config.Config.block_sep + keya, valuea)
    c.set_value(block2 + config.Config.block_sep + keyb, valueb)

    d = c.subblock_view(block1)

    if not d.has_value(keya):
        test_error("Subblock does not inherit expected keys")

    if d.has_value(keyb):
        test_error("Subblock inherited unrelated key")

    c.set_value(block1 + config.Config.block_sep + keya, valueb)

    get_valuea1 = d.get_value(keya)

    if not valueb == get_valuea1:
        test_error("Subblock view persisted a changed value")

    d.set_value(keya, valuea)

    get_valuea2 = d.get_value(keya)

    if not valuea == get_valuea2:
        test_error("Subblock view set value was not changed in parent")


def test_merge_config():
    from sprokit.pipeline import config

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

    if not valuea == get_valuea:
        test_error("Unmerged key changed")

    get_valueb = c.get_value(keyb)

    if not valueb == get_valueb:
        test_error("Conflicting key was not overwritten")

    get_valuec = c.get_value(keyc)

    if not valuec == get_valuec:
        test_error("New key did not appear")


def test_dict():
    from sprokit.pipeline import config

    c = config.empty_config()

    key = 'key'
    value = 'oldvalue'

    if key in c:
        test_error("'%s' is in an empty config" % key)

    if c:
        test_error("An empty config is not falsy")

    c[key] = value

    if not c[key] == value:
        test_error("Value was not set")

    if key not in c:
        test_error("'%s' is not in config after insertion" % key)

    if not len(c) == 1:
        test_error("The len() operator is incorrect")

    if not c:
        test_error("A non-empty config is not truthy")

    value = 'newvalue'
    origvalue = 'newvalue'

    c[key] = value

    value = 'replacedvalue'

    if not c[key] == origvalue:
        test_error("Value was overwritten")

    del c[key]

    expect_exception('getting an unset value', BaseException,
                     c.__getitem__, key)

    expect_exception('deleting an unset value', BaseException,
                     c.__delitem__, key)

    value = 10

    c[key] = value

    if not c[key] == str(value):
        test_error("Value was not converted to a string")


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    from sprokit.test.test import *

    run_test(testname, find_tests(locals()))
