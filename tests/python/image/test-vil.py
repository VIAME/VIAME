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
        import vistk.image.vil
    except:
        log("Error: Failed to import the vil module")


def test_create():
    from vistk.image import vil

    try:
        config.empty_config()
    except:
        log("Error: Failed to create an empty configuration")

    config.ConfigKey()
    config.ConfigKeys()
    config.ConfigValue()


def test_vil_to_numpy():
    from vistk.image import vil
    import numpy

    log("Error: Not implemented")


def test_numpy_to_vil():
    from vistk.image import vil
    import numpy

    log("Error: Not implemented")


def test_datum():
    from vistk.image import vil
    import numpy

    log("Error: Not implemented")


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'vil_to_numpy':
        test_vil_to_numpy()
    elif testname == 'numpy_to_vil':
        test_numpy_to_vil()
    elif testname == 'datum':
        test_datum()
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
