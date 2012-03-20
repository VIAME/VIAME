#!/usr/bin/env python
#ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def log(msg):
    import sys
    sys.stderr.write("%s\n" % msg)


def test_import():
    try:
        import vistk.pipeline.stamp
    except:
        log("Error: Failed to import the stamp module")


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
        log("Error: New stamps have the same color")

    if not s.is_same_color(sc):
        log("Error: Copied stamps do not have the same color")

    if s > si:
        log("Error: A stamp is greater than its increment")

    if si < s:
        log("Error: A stamp is greater than its increment")

    if s < t or t < s:
        log("Error: Different colored stamps return True for comparison")


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'api_calls':
        test_api_calls()
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
