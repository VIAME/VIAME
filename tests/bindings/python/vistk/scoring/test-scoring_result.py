#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.scoring.scoring_result
    except:
        test_error("Failed to import the scoring_result module")


def test_api_calls():
    from vistk.scoring import scoring_result

    result = scoring_result.new_result(1, 1, 1)
    result = scoring_result.new_result(1, 1, 1, 1)

    result.true_positives()
    result.false_positives()
    result.total_trues()
    result.total_possible()
    result.percent_detection()
    result.precision()
    result.specificity()

    result + result


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
        { 'import': test_import
        , 'api_calls': test_api_calls
        }

    from vistk.test.test import *

    run_test(testname, tests)
