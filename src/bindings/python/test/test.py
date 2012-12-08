#ckwg +4
# Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_error(msg):
    import sys

    sys.stderr.write("Error: %s\n" % msg)


def expect_exception(action, kind, func, *args):
    got_exception = False

    try:
        func(*args)
    except kind:
        got_exception = True
    except BaseException as e:
        test_error("Got unexpected exception: %s" % str(e))

        got_exception = True
    except:
        test_error("Got non-standard exception")

        got_exception = True

    if not got_exception:
        test_error("Did not get exception when %s" % action)


def run_test(testname, tests, *args, **kwargs):
    if testname not in tests:
        import sys

        test_error("No such test '%s'" % testname)

        sys.exit(1)

    try:
        tests[testname](*args, **kwargs)
    except BaseException as e:
        test_error("Unexpected exception: %s" % str(e))
