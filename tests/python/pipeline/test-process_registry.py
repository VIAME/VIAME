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

            self.ran_init = False
            self.ran_step = False
            self.ran_constraints = False
            self.ran_connect_input_port = False
            self.ran_connect_output_port = False
            self.ran_input_ports = False
            self.ran_output_ports = False
            self.ran_input_port_info = False
            self.ran_output_port_info = False
            self.ran_available_config = False
            self.ran_conf_info = False

        def _init(self):
            self.ran_init = True

            self._base_init()

        def _step(self):
            self.ran_step = True

            self.check()

            self._base_step()

        def _constraints(self):
            self.ran_step = True

            return self._base_constraints()

        def _connect_input_port(self, port, edge):
            self.ran_connect_input_port = True

            self._base_connect_input_port(port, edge)

        def _connect_output_port(self, port, edge):
            self.ran_connect_output_port = True

            self._base_connect_output_port(port, edge)

        def _input_ports(self):
            self.ran_input_ports = True

            return self._base_input_ports()

        def _output_ports(self):
            self.ran_output_ports = True

            return self._base_output_ports()

        def _input_port_info(self, port):
            self.ran_input_port_info = True

            return self._base_input_port_info(port)

        def _output_port_info(self, port):
            self.ran_output_port_info = True

            return self._base_output_port_info(port)

        def _available_config(self):
            self.ran_available_config = True

            return self._base_available_config()

        def _config_info(self, key):
            self.ran_conf_info = True

            return self._base_conf_info(key)

        def check(self):
            if not self.ran_init:
                log("Error: _init override was not called")
            if not self.ran_step:
                log("Error: _step override was not called")
            if not self.ran_constraints:
                log("Error: _constraints override was not called")
            if not self.ran_connect_input_port:
                log("Error: _connect_input_port override was not called")
            if not self.ran_connect_output_port:
                log("Error: _connect_output_port override was not called")
            if not self.ran_input_ports:
                log("Error: _input_ports override was not called")
            if not self.ran_output_ports:
                log("Error: _output_ports override was not called")
            if not self.ran_input_port_info:
                log("Error: _input_port_info override was not called")
            if not self.ran_output_port_info:
                log("Error: _output_port_info override was not called")
            if not self.ran_available_config:
                log("Error: _available_config override was not called")
            if not self.ran_conf_info:
                log("Error: _conf_info override was not called")

    return PythonExample


def base_example_process():
    from vistk.pipeline import process

    class PythonBaseExample(process.PythonProcess):
        def __init__(self, conf):
            process.PythonProcess.__init__(self, conf)

        def check(self):
            pass

    return PythonBaseExample


def test_register():
    from vistk.pipeline import config
    from vistk.pipeline import process
    from vistk.pipeline import process_registry

    proc_type = 'python_example'
    proc_desc = 'simple description'

    reg = process_registry.ProcessRegistry.self()

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

    proc_type = 'python_example'
    proc_desc = 'simple description'

    proc_base_type = 'python_base_example'
    proc_base_desc = 'simple base description'

    iport = 'no_such_iport'
    oport = 'no_such_oport'
    key = 'no_such_key'

    reg = process_registry.ProcessRegistry.self()

    reg.register_process(proc_type, proc_desc, example_process())
    reg.register_process(proc_base_type, proc_base_desc, base_example_process())

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

        p.check()

    p = reg.create_process(proc_type, c)
    check_process(p)

    p = reg.create_process(proc_base_type, c)
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

    try:
        main(testname)
    except BaseException as e:
        log("Error: Unexpected exception: %s" % str(e))
