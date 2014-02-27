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


def find_tests(scope):
    prefix = 'test_'
    tests = {}

    for name, obj in scope.items():
        if name.startswith(prefix) and callable(obj):
            tests[name[len(prefix):]] = obj

    return tests


def test_error(msg):
    import sys

    sys.stderr.write("Error: %s\n" % msg)


def expect_exception(action, kind, func, *args, **kwargs):
    got_exception = False

    try:
        func(*args, **kwargs)
    except kind:
        import sys

        t = sys.exc_info()[0]
        e = sys.exc_info()[1]

        sys.stderr.write("Got expected exception: %s: %s\n" % (str(t.__name__), str(e)))

        got_exception = True
    except BaseException:
        import sys
        import traceback

        t = sys.exc_info()[0]
        e = sys.exc_info()[1]
        bt = sys.exc_info()[2]
        bt_str = ''.join(traceback.format_tb(bt))

        test_error("Got unexpected exception: %s: %s:\n%s" % (str(t.__name__), str(e), bt_str))

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
    except BaseException:
        import sys
        import traceback

        e = sys.exc_info()[1]
        bt = sys.exc_info()[2]
        bt_str = ''.join(traceback.format_tb(bt))

        test_error("Unexpected exception: %s:\n%s" % (str(e), bt_str))
