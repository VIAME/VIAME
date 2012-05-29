#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_error(msg):
    import sys
    sys.stderr.write('Error: %s\n' % msg)


def expect_exception(action, kind, func, *args):
    got_exception = False

    try:
        func(*args)
    except kind:
        got_exception = True
    except:
        pass

    if not got_exception:
        test_error('Did not get exception when %s' % action)
