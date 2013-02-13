#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        import vistk.modules.modules
    except:
        test_error("Failed to import the modules module")


def test_load():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    types = reg.types()

    if 'test_python_process' not in types:
        test_error("Failed to load Python processes")


def test_masking():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    types = reg.types()

    if 'test_python_process' in types:
        test_error("Failed to mask out Python processes")


def test_extra_modules():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    types = reg.types()

    if 'extra_test_python_process' not in types:
        test_error("Failed to load extra Python processes")


def test_pythonpath():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry
    from vistk.pipeline import scheduler_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    types = reg.types()

    if 'pythonpath_test_process' not in types:
        test_error("Failed to load extra Python processes accessible from PYTHONPATH")

    reg = scheduler_registry.SchedulerRegistry.self()

    types = reg.types()

    if 'pythonpath_test_scheduler' not in types:
        test_error("Failed to load extra Python schedulers accessible from PYTHONPATH")


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
        , 'load': test_load
        , 'masking': test_masking
        , 'extra_modules': test_extra_modules
        , 'pythonpath': test_pythonpath
        }

    from vistk.test.test import *

    run_test(testname, tests)
