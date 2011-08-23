#!/usr/bin/env python
#ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def log(msg):
    import sys
    sys.stderr.write("%s\n" % msg)


def ensure_exception(action, func, *args):
    got_exception = False

    try:
        func(*args)
    except:
        got_exception = True

    if not got_exception:
        log("Error: Did not get exception when %s" % action)


def test_import():
    try:
        import vistk.pipeline.config
    except:
        log("Error: Failed to import the config module")


def test_create():
    from vistk.pipeline import config

    try:
        config.empty_config()
    except:
        log("Error: Failed to create an empty configuration")

    config.ConfigKey()
    config.ConfigKeys()
    config.ConfigValue()


def test_has_value():
    from vistk.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valuea = 'value_a'

    c.set_value(keya, valuea)

    if not c.has_value(keya):
        log("Error: Block does not have value which was set")

    if c.has_value(keyb):
        log("Error: Block has value which was not set")


def test_get_value():
    from vistk.pipeline import config

    c = config.empty_config()

    keya = 'keya'

    valuea = 'value_a'

    c.set_value(keya, valuea)

    get_valuea = c.get_value(keya)

    if not valuea == get_valuea:
        log("Error: Did not retrieve value that was set")


def test_get_value_no_exist():
    from vistk.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valueb = 'value_b'

    ensure_exception("retrieving an unset value",
                     c.get_value, keya)

    get_valueb = c.get_value(keyb, valueb)

    if not valueb == get_valueb:
        log("Error: Did not retrieve default when requesting unset value")


def test_unset_value():
    from vistk.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valuea = 'value_a'
    valueb = 'value_b'

    c.set_value(keya, valuea)
    c.set_value(keyb, valueb)

    c.unset_value(keya)

    ensure_exception("retrieving an unset value",
                     c.get_value, keya)

    get_valueb = c.get_value(keyb)

    if not valueb == get_valueb:
        log("Error: Did not retrieve value when requesting after an unrelated unset")


def test_available_values():
    from vistk.pipeline import config

    c = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'

    valuea = 'value_a'
    valueb = 'value_b'

    c.set_value(keya, valuea)
    c.set_value(keyb, valueb)

    avail = c.available_values()

    if not len(avail) == 2:
        log("Error: Did not retrieve correct number of keys")

    try:
        for val in avail:
            pass
    except:
        log("Error: Available values is not iterable")


def test_read_only():
    from vistk.pipeline import config

    c = config.empty_config()

    keya = 'keya'

    valuea = 'value_a'
    valueb = 'value_b'

    c.set_value(keya, valuea)

    c.mark_read_only(keya)

    ensure_exception("setting a read only value",
                     c.set_value, keya, valueb)

    get_valuea = c.get_value(keya)

    if not valuea == get_valuea:
        log("Error: Read only value changed")


def test_read_only_unset():
    from vistk.pipeline import config

    c = config.empty_config()

    keya = 'keya'

    valuea = 'value_a'

    c.set_value(keya, valuea)

    c.mark_read_only(keya)

    ensure_exception("unsetting a read only value",
                     c.unset_value, keya)

    get_valuea = c.get_value(keya)

    if not valuea == get_valuea:
        log("Error: Read only value was unset")


def test_subblock():
    from vistk.pipeline import config

    c = config.empty_config()

    block1 = 'block1'
    block2 = 'block2'

    keya = 'keya'
    keyb = 'keyb'
    keyc = 'keyc'

    valuea = 'value_a'
    valueb = 'value_b'
    valuec = 'value_c'

    c.set_value(block1 + config.Config.block_sep + keya, valuea)
    c.set_value(block1 + config.Config.block_sep + keyb, valueb)
    c.set_value(block2 + config.Config.block_sep + keyc, valuec)

    d = c.subblock(block1)

    get_valuea = d.get_value(keya)

    if not valuea == get_valuea:
        log("Error: Subblock does not inherit expected keys")

    get_valueb = d.get_value(keyb)

    if not valueb == get_valueb:
        log("Error: Subblock does not inherit expected keys")

    if d.has_value(keyc):
        log("Error: Subblock inherited unrelated key")


def test_subblock_view():
    from vistk.pipeline import config

    c = config.empty_config()

    block1 = 'block1'
    block2 = 'block2'

    keya = 'keya'
    keyb = 'keyb'
    keyc = 'keyc'

    valuea = 'value_a'
    valueb = 'value_b'
    valuec = 'value_c'

    c.set_value(block1 + config.Config.block_sep + keya, valuea)
    c.set_value(block2 + config.Config.block_sep + keyb, valueb)

    d = c.subblock_view(block1)

    if not d.has_value(keya):
        log("Error: Subblock does not inherit expected keys")

    if d.has_value(keyb):
        log("Error: Subblock inherited unrelated key")

    c.set_value(block1 + config.Config.block_sep + keya, valueb)

    get_valuea1 = d.get_value(keya)

    if not valueb == get_valuea1:
        log("Error: Subblock view persisted a changed value")

    d.set_value(keya, valuea)

    get_valuea2 = d.get_value(keya)

    if not valuea == get_valuea2:
        log("Error: Subblock view set value was not changed in parent")


def test_merge_config():
    from vistk.pipeline import config

    c = config.empty_config()
    d = config.empty_config()

    keya = 'keya'
    keyb = 'keyb'
    keyc = 'keyc'

    valuea = 'value_a'
    valueb = 'value_b'
    valuec = 'value_c'

    c.set_value(keya, valuea)
    c.set_value(keyb, valuea)

    d.set_value(keyb, valueb)
    d.set_value(keyc, valuec)

    c.merge_config(d)

    get_valuea = c.get_value(keya)

    if not valuea == get_valuea:
        log("Error: Unmerged key changed")

    get_valueb = c.get_value(keyb)

    if not valueb == get_valueb:
        log("Error: Conflicting key was not overwritten")

    get_valuec = c.get_value(keyc)

    if not valuec == get_valuec:
        log("Error: New key did not appear")


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'has_value':
        test_has_value()
    elif testname == 'get_value':
        test_get_value()
    elif testname == 'get_value_no_exist':
        test_get_value_no_exist()
    elif testname == 'unset_value':
        test_unset_value()
    elif testname == 'available_values':
        test_available_values()
    elif testname == 'read_only':
        test_read_only()
    elif testname == 'read_only_unset':
        test_read_only_unset()
    elif testname == 'subblock':
        test_subblock()
    elif testname == 'subblock_view':
        test_subblock_view()
    elif testname == 'merge_config':
        test_merge_config()
    else:
        log("Error: No such test '%s'" % testname)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        log("Error: Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    try:
        main(testname)
    except BaseException as e:
        log("Error: Unexpected exception: %s" % str(e))
