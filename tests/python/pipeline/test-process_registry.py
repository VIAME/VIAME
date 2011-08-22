#!/usr/bin/env python
#ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def log(msg):
    import sys
    sys.stderr.write("%s\n" % msg)


def test_import():
    try:
        import vistk.pipeline.process_registry
    except:
        log("Error: Failed to import the process_registry module")


def test_create():
    from vistk.pipeline import process_registry

    process_registry.ProcessRegistry.self()
    process_registry.ProcessType()
    process_registry.ProcessTypes()
    process_registry.ProcessDescription()


def test_api_calls():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    proc_type = 'orphan'
    c = config.empty_config()

    reg.create_process(proc_type, c)
    reg.types()
    reg.description(proc_type)


def example_process():
    from vistk.pipeline import process

    class PythonExample(process.PythonProcess):
        def __init__(self, conf):
            process.PythonProcess.__init__(self, conf)

    return PythonExample


def test_register():
    from vistk.pipeline import config
    from vistk.pipeline import process
    from vistk.pipeline import process_registry

    reg = process_registry.ProcessRegistry.self()

    proc_type = 'python_example'
    proc_desc = 'simple description'

    reg.register_process(proc_type, proc_desc, example_process())

    if not proc_desc == reg.description(proc_type):
        log("Error: Description was not preserved when registering")

    c = config.empty_config()

    try:
        reg.create_process(proc_type, c)
    except:
        log("Error: Could not create newly registered process type")


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'api_calls':
        test_api_calls()
    elif testname == 'register':
        test_register()
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
