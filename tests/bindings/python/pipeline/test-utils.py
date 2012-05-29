#!@PYTHON_EXECUTABLE@
#ckwg +5
# Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.pipeline.utils
    except:
        test_error("Failed to import the modules module")


def test_create():
    from vistk.pipeline import utils

    utils.ThreadName()


def test_name_thread():
    from vistk.pipeline import utils

    utils.name_thread("a_name")


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'name_thread':
        test_name_thread()
    else:
        test_error("No such test '%s'" % testname)


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    from vistk.test.test import *

    try:
        main(testname)
    except BaseException as e:
        test_error("Unexpected exception: %s" % str(e))
