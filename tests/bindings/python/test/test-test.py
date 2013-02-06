#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_return_code():
    import sys

    sys.exit(1)


def test_error_string():
    test_error('an error')


def test_error_string_mid():
    import sys

    sys.stderr.write('Test')
    test_error('an error')


def test_error_string_stdout():
    import sys

    sys.stdout.write('Error: an error\n')


def test_error_string_second_line():
    import sys

    sys.stderr.write('Not an error\n')
    test_error("an error")


def raise_exception():
    raise NotImplementedError


def test_expected_exception():
    expect_exception('when throwing an exception', NotImplementedError,
                     raise_exception)


def test_unexpected_exception():
    expect_exception('when throwing an unexpected exception', SyntaxError,
                     raise_exception)


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
        { 'return_code': test_return_code
        , 'error_string': test_error_string
        , 'error_string_mid': test_error_string_mid
        , 'error_string_stdout': test_error_string_stdout
        , 'error_string_second_line': test_error_string_second_line
        , 'expected_exception': test_expected_exception
        , 'unexpected_exception': test_unexpected_exception
        }

    from vistk.test.test import *

    run_test(testname, tests)
