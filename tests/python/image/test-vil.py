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
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 3

    shape = (width, height, planes)

    types = [ (test_image.make_image_bool, np.bool)
            , (test_image.make_image_uint8_t, np.uint8)
            , (test_image.make_image_float, np.float32)
            , (test_image.make_image_double, np.double)
            ]

    for f, t in types:
        i = f(width, height, planes)

        if not i.dtype == t:
            log("Error: Wrong type returned: got: '%s' expected: '%s'" % (i.dtype, t))

        if not i.ndim == 3:
            log("Error: Did not get a 3-dimensional array: got: '%d' expected: '%d'" % (i.dim, 3))

        if not i.shape == shape:
            log("Error: Did not get expected array sizes: got '%s' expected: '%s'" % (str(i.shape), str(shape)))


def test_numpy_to_vil():
    from vistk.image import vil
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 3

    shape = (width, height, planes)

    size = width * height * planes

    types = [ (test_image.take_image_bool, np.bool)
            , (test_image.take_image_uint8_t, np.uint8)
            , (test_image.take_image_float, np.float32)
            , (test_image.take_image_double, np.double)
            ]

    for f, t in types:
        i = np.zeros(shape, dtype=t)

        sz = f(i)

        if not sz == size:
            log("Error: Wrong size calculated: got: '%d' expected: '%d'" % (sz, size))


def test_datum():
    from vistk.image import vil
    from vistk.test import test_image
    import numpy as np

    width = 800
    height = 600
    planes = 3

    shape = (width, height, planes)

    types = [ (test_image.save_image_bool, 'bool', np.bool)
            , (test_image.save_image_uint8_t, 'byte', np.uint8)
            # \todo How to save these?
            , (test_image.save_image_float, 'float', np.float32)
            , (test_image.save_image_double, 'double', np.double)
            ]

    for f, pt, t in types:
        a = np.zeros(shape, dtype=t)

        if not f(a, '%s.tiff' % pt):
            log("Error: Failed to save %s image" % pt)


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
