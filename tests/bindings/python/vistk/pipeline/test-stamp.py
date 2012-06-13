#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.pipeline.stamp
    except:
        test_error("Failed to import the stamp module")


def test_create():
    from vistk.pipeline import stamp

    stamp.new_stamp()


def test_api_calls():
    from vistk.pipeline import stamp

    s = stamp.new_stamp()
    sc = stamp.copied_stamp(s)
    si = stamp.incremented_stamp(s)
    t = stamp.new_stamp()
    sr = stamp.recolored_stamp(s, t)

    if s.is_same_color(t):
        test_error("New stamps have the same color")

    if not s.is_same_color(sc):
        test_error("Copied stamps do not have the same color")

    if s > si:
        test_error("A stamp is greater than its increment")

    if si < s:
        test_error("A stamp is greater than its increment")

    if s < t or t < s:
        test_error("Different colored stamps return True for comparison")


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'api_calls':
        test_api_calls()
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
