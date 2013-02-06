#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.pipeline.utils
    except:
        test_error("Failed to import the utils module")


def test_create():
    from vistk.pipeline import utils

    utils.ThreadName()


def test_name_thread():
    from vistk.pipeline import utils

    utils.name_thread("a_name")


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
        , 'name_thread': test_name_thread
        }

    from vistk.test.test import *

    run_test(testname, tests)
