#!/usr/bin/env python
#ckwg +28
# Copyright 2012-2013 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# TEST_PROPERTY(WILL_FAIL, TRUE)
def test_return_code():
    import sys

    sys.exit(1)


# TEST_PROPERTY(WILL_FAIL, TRUE)
def test_error_string():
    test_error('an error')


def test_error_string_mid():
    import sys

    sys.stderr.write('Test')
    test_error('an error')


# TEST_PROPERTY(WILL_FAIL, TRUE)
def test_error_string_stdout():
    import sys

    sys.stdout.write('Error: an error\n')


# TEST_PROPERTY(WILL_FAIL, TRUE)
def test_error_string_second_line():
    import sys

    sys.stderr.write('Not an error\n')
    test_error("an error")


def raise_exception():
    raise NotImplementedError


def test_expected_exception():
    expect_exception('when throwing an exception', NotImplementedError,
                     raise_exception)


# TEST_PROPERTY(WILL_FAIL, TRUE)
def test_unexpected_exception():
    expect_exception('when throwing an unexpected exception', SyntaxError,
                     raise_exception)


# TEST_PROPERTY(ENVIRONMENT, TEST_ENVVAR=test_value)
def test_environment():
    import os

    envvar = 'TEST_ENVVAR'

    if envvar not in os.environ:
        test_error('failed to get environment from CTest')
    else:
        expected = 'test_value'

        envvalue = os.environ[envvar]

        if envvalue != expected:
            test_error('did not get expected value')


if __name__ == '__main__':
    import os
    import sys

    if not len(sys.argv) == 4:
        test_error("Expected three arguments")
        sys.exit(1)

    testname = sys.argv[1]

    os.chdir(sys.argv[2])

    sys.path.append(sys.argv[3])

    from sprokit.test.test import *

    run_test(testname, find_tests(locals()))
