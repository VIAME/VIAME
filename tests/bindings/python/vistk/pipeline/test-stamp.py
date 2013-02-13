#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.pipeline.stamp
    except:
        test_error("Failed to import the stamp module")


def test_create():
    from vistk.pipeline import stamp

    stamp.new_stamp(1)


def test_api_calls():
    from vistk.pipeline import stamp

    s = stamp.new_stamp(1)
    si = stamp.incremented_stamp(s)
    t = stamp.new_stamp(2)

    if s > si:
        test_error("A stamp is greater than its increment")

    if si < s:
        test_error("A stamp is greater than its increment")

    si2 = stamp.incremented_stamp(si)
    ti = stamp.incremented_stamp(t)

    if not si2 == ti:
        test_error("Stamps with different rates do not compare as equal")


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
        , 'create': test_create
        , 'api_calls': test_api_calls
        }

    from vistk.test.test import *

    run_test(testname, tests)
