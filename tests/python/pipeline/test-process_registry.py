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


def test_wrapper_api():
    from vistk.pipeline import config
    from vistk.pipeline import edge
    from vistk.pipeline import process
    from vistk.pipeline import process_registry

    reg = process_registry.ProcessRegistry.self()

    proc_type = 'python_example'
    proc_desc = 'simple description'

    iport = 'no_such_iport'
    oport = 'no_such_oport'
    key = 'no_such_key'

    reg.register_process(proc_type, proc_desc, example_process())

    c = config.empty_config()

    def check_process(p):
        p.input_ports()
        p.output_ports()
        ensure_exception("asking for info on a non-existant input port",
                         p.input_port_info, iport)
        ensure_exception("asking for info on a non-existant output port",
                         p.output_port_info, oport)

        e = edge.Edge(c)

        ensure_exception("connecting to a non-existant input port",
                         p.connect_input_port, iport, e)
        ensure_exception("connecting to a non-existant output port",
                         p.connect_output_port, oport, e)

        p.available_config()
        ensure_exception("asking for info on a non-existant config key",
                         p.config_info, key)

        p.init()
        p.step()

    p = reg.create_process(proc_type, c)
    check_process(p)


def main(testname):
    if testname == 'import':
        test_import()
    elif testname == 'create':
        test_create()
    elif testname == 'api_calls':
        test_api_calls()
    elif testname == 'register':
        test_register()
    elif testname == 'wrapper_api':
        test_wrapper_api()
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
