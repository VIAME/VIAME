#!/usr/bin/env python
#ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def raise_exception():
    raise NotImplementedError


def main(testname):
    if testname == 'return_code':
        import sys
        sys.exit(1)
    elif testname == 'error_string':
        test_error("an error")
    elif testname == 'error_string_mid':
        import sys
        sys.stderr.write('Test')
        test_error("an error")
    elif testname == 'error_string_stdout':
        import sys
        sys.stdout.write("Error: an error\n")
    elif testname == 'error_string_second_line':
        import sys
        sys.stderr.write('Not an error\n')
        test_error("an error")
    elif testname == 'expected_exception':
        expect_exception('when throwing an exception', NotImplementedError,
                         raise_exception)
    elif testname == 'unexpected_exception':
        expect_exception('when throwing an unexpected exception', SyntaxError,
                         raise_exception)
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
