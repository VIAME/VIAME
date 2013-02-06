#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.utilities.timestamp
    except:
        test_error("Failed to import the timestamp module")


def test_api_calls():
    from vistk.utilities import timestamp

    # TODO Check to make sure the correct ctor is called.

    t = timestamp.Timestamp()
    t = timestamp.Timestamp(1)
    t = timestamp.Timestamp(1.0)
    t = timestamp.Timestamp(1.0, 0)

    t.has_time()
    t.has_frame()
    t.is_valid()

    t == t
    t <  t
    t >  t


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    tests = \
        { 'import': test_import
        , 'api_calls': test_api_calls
        }

    from vistk.test.test import *

    run_test(testname, tests)
