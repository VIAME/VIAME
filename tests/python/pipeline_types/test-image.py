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
        import vistk.pipeline_types.image
    except:
        log("Error: Failed to import the image module")


def test_api_calls():
    from vistk.pipeline_types import image

    image.t_bitmask
    image.t_bytemask
    image.t_byte_grayscale
    image.t_float_grayscale
    image.t_byte_rgb
    image.t_float_rgb
    image.t_byte_bgr
    image.t_float_bgr
    image.t_byte_rgba
    image.t_float_rgba
    image.t_byte_bgra
    image.t_float_bgra
    image.t_byte_yuv
    image.t_float_yuv
    image.t_ipl


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
