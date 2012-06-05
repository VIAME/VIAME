#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.utilities.homography
    except:
        test_error("Failed to import the homography module")


def test_api_calls():
    from vistk.utilities import homography
    from vistk.utilities import timestamp

    homog = homography.ImageToImageHomography()

    t = homog.transform()
    homog.is_valid()
    homog.is_new_reference()
    homog.set_transform(t)
    homog.set_identity()
    homog.set_valid(False)
    homog.set_new_reference(False)

    t.get(0, 0)
    t.set(0, 0, 1)

    homog == homog

    s = timestamp.Timestamp()
    d = timestamp.Timestamp()

    homog.inverse()
    homog.set_source(s)
    homog.set_destination(d)
    homog.source()
    homog.destination()


def main(testname):
    if testname == 'import':
        test_import()
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
