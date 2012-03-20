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
        import vistk.modules.modules
    except:
        log("Error: Failed to import the modules module")


def test_load():
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    types = reg.types()

    if 'test_python_process' not in types:
        log("Error: Failed to load Python processes")


def test_masking():
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    types = reg.types()

    if 'test_python_process' in types:
        log("Error: Failed to mask out Python processes")


def test_extra_modules():
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    types = reg.types()

    if 'extra_test_python_process' not in types:
        log("Error: Failed to load extra Python processes")


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'load':
        test_load()
    elif testname == 'masking':
        test_masking()
    elif testname == 'extra_modules':
        test_extra_modules()
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

    main(testname)
    #try:
        #main(testname)
    #except BaseException as e:
        #log("Error: Unexpected exception: %s" % str(e))
