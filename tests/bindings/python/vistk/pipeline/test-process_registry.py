#!@PYTHON_EXECUTABLE@
#ckwg +4
# Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


def test_import():
    try:
        from vistk.pipeline import config
        import vistk.pipeline.process_registry
    except:
        test_error("Failed to import the process_registry module")


def test_create():
    from vistk.pipeline import config
    from vistk.pipeline import process_registry

    process_registry.ProcessRegistry.self()
    process_registry.ProcessDescription()
    process_registry.ProcessModule()


def test_api_calls():
    from vistk.pipeline import config
    from vistk.pipeline import modules
    from vistk.pipeline import process
    from vistk.pipeline import process_registry

    modules.load_known_modules()

    reg = process_registry.ProcessRegistry.self()

    proc_type = 'orphan'
    c = config.empty_config()

    reg.create_process(proc_type, process.ProcessName())
    reg.create_process(proc_type, process.ProcessName(), c)
    reg.types()
    reg.description(proc_type)

    process_registry.Process.constraint_no_threads
    process_registry.Process.constraint_no_reentrancy
    process_registry.Process.constraint_unsync_input
    process_registry.Process.constraint_unsync_output
    process_registry.Process.port_heartbeat
    process_registry.Process.config_name
    process_registry.Process.config_type
    process_registry.Process.type_any
    process_registry.Process.type_none
    process_registry.Process.type_data_dependent
    process_registry.Process.type_flow_dependent
    process_registry.Process.flag_output_const
    process_registry.Process.flag_input_static
    process_registry.Process.flag_input_mutable
    process_registry.Process.flag_input_nodep
    process_registry.Process.flag_required


def example_process():
    from vistk.pipeline import process

    class PythonExample(process.PythonProcess):
        def __init__(self, conf):
            process.PythonProcess.__init__(self, conf)

            self.ran_configure = False
            self.ran_init = False
            self.ran_reset = False
            self.ran_step = False
            self.ran_constraints = False
            self.ran_input_ports = False
            self.ran_output_ports = False
            self.ran_input_port_info = False
            self.ran_output_port_info = False
            self.ran_set_input_port_type = False
            self.ran_set_output_port_type = False
            self.ran_available_config = False
            self.ran_conf_info = False

        def _configure(self):
            self.ran_configure = True

            self._base_configure()

        def _init(self):
            self.ran_init = True

            self._base_init()

        def _reset(self):
            self.ran_reset = True

            self._base_reset()

        def _step(self):
            self.ran_step = True

            self._base_step()

        def _constraints(self):
            self.ran_constraints = True

            return self._base_constraints()

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

        def _set_input_port_type(self, port, type):
            self.ran_set_input_port_type = True

            return self._base_set_input_port_type(port, type)

        def _set_output_port_type(self, port, type):
            self.ran_set_output_port_type = True

            return self._base_set_output_port_type(port, type)

        def _available_config(self):
            self.ran_available_config = True

            return self._base_available_config()

        def _config_info(self, key):
            self.ran_conf_info = True

            return self._base_config_info(key)

        def check(self):
            if not self.ran_configure:
                test_error("_configure override was not called")
            if not self.ran_init:
                test_error("_init override was not called")
            if not self.ran_reset:
                test_error("_reset override was not called")
            # TODO: See TODO below.
            #if not self.ran_step:
            #    test_error("_step override was not called")
            if not self.ran_constraints:
                test_error("_constraints override was not called")
            if not self.ran_input_ports:
                test_error("_input_ports override was not called")
            if not self.ran_output_ports:
                test_error("_output_ports override was not called")
            if not self.ran_input_port_info:
                test_error("_input_port_info override was not called")
            if not self.ran_output_port_info:
                test_error("_output_port_info override was not called")
            if not self.ran_set_input_port_type:
                test_error("_set_input_port_type override was not called")
            if not self.ran_set_output_port_type:
                test_error("_set_output_port_type override was not called")
            if not self.ran_available_config:
                test_error("_available_config override was not called")
            if not self.ran_conf_info:
                test_error("_conf_info override was not called")

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
        test_error("Description was not preserved when registering")

    try:
        p = reg.create_process(proc_type, process.ProcessName())
        if p is None:
            raise Exception()
    except:
        test_error("Could not create newly registered process type")


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
    ptype = 'no_type'

    reg = process_registry.ProcessRegistry.self()

    reg.register_process(proc_type, proc_desc, example_process())
    reg.register_process(proc_base_type, proc_base_desc, base_example_process())

    def check_process(p):
        if p is None:
            test_error("Got a 'None' process")
            return

        p.constraints()

        p.input_ports()
        p.output_ports()
        expect_exception("asking for info on a non-existant input port", RuntimeError,
                         p.input_port_info, iport)
        expect_exception("asking for info on a non-existant output port", RuntimeError,
                         p.output_port_info, oport)

        e = edge.Edge()

        expect_exception("connecting to a non-existant input port", RuntimeError,
                         p.connect_input_port, iport, e)
        expect_exception("connecting to a non-existant output port", RuntimeError,
                         p.connect_output_port, oport, e)

        p.available_config()
        expect_exception("asking for info on a non-existant config key", RuntimeError,
                         p.config_info, key)

        expect_exception("setting a type on a non-existent input port", RuntimeError,
                         p.set_input_port_type, iport, ptype)
        expect_exception("setting a type on a non-existent output port", RuntimeError,
                         p.set_output_port_type, oport, ptype)

        p.reset()

        p.configure()
        p.init()
        # TODO: Can't check this because the core frequency of the process
        # cannot be set. Needs to be stepped within a pipeline to verify this.
        # Enable the ran_step check in p.check when this is fixed.
        #p.step()

        p.check()

    p = reg.create_process(proc_type, process.ProcessName())
    check_process(p)

    p = reg.create_process(proc_base_type, process.ProcessName())
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
