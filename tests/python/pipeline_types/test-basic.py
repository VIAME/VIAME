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
        import vistk.pipeline_types.basic
    except:
        log("Error: Failed to import the basic module")


def test_api_calls():
    from vistk.pipeline_types import basic

    basic.t_bool
    basic.t_char
    basic.t_string
    basic.t_integer
    basic.t_unsigned
    basic.t_float
    basic.t_double
    basic.t_byte
    basic.t_vec_char
    basic.t_vec_string
    basic.t_vec_integer
    basic.t_vec_unsigned
    basic.t_vec_float
    basic.t_vec_double
    basic.t_vec_byte


def main(testname):
    if testname == 'import':
        test_import()
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
