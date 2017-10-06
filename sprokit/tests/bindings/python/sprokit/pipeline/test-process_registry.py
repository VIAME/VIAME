#!/usr/bin/env python
#ckwg +28
# Copyright 2011-2013 by Kitware, Inc.
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


def test_import():
    try:
        from sprokit.pipeline import config
        import sprokit.pipeline.process_factory
    except:
        test_error("Failed to import the process_factory module")


def test_create():
    from sprokit.pipeline import config
    from sprokit.pipeline import process_factory

    process_factory.ProcessDescription()
    process_factory.ProcessModule()


def test_api_calls():
    from sprokit.pipeline import config
    from sprokit.pipeline import modules
    from sprokit.pipeline import process
    from sprokit.pipeline import process_factory

    modules.load_known_modules()

    proc_type = 'orphan'
    c = config.empty_config()

    process_factory.create_process(proc_type, process.ProcessName())
    process_factory.create_process(proc_type, process.ProcessName(), c)
    process_factory.types()
    process_factory.description(proc_type)

    process_factory.Process.property_no_threads
    process_factory.Process.property_no_reentrancy
    process_factory.Process.property_unsync_input
    process_factory.Process.property_unsync_output
    process_factory.Process.port_heartbeat
    process_factory.Process.config_name
    process_factory.Process.config_type
    process_factory.Process.type_any
    process_factory.Process.type_none
    process_factory.Process.type_data_dependent
    process_factory.Process.type_flow_dependent
    process_factory.Process.flag_output_const
    process_factory.Process.flag_input_static
    process_factory.Process.flag_input_mutable
    process_factory.Process.flag_input_nodep
    process_factory.Process.flag_required

    cluster_bases = process_factory.ProcessCluster.__bases__
    if not cluster_bases[0] == process_factory.Process:
        test_error("The cluster class does not inherit from the process class")


def example_process(check_init):
    from sprokit.pipeline import process

    class PythonExample(process.PythonProcess):
        def __init__(self, conf):
            process.PythonProcess.__init__(self, conf)

            self.ran_configure = check_init
            self.ran_init = check_init
            self.ran_reset = check_init
            self.ran_step = check_init
            self.ran_reconfigure = check_init
            self.ran_properties = check_init
            self.ran_input_ports = check_init
            self.ran_output_ports = check_init
            self.ran_input_port_info = check_init
            self.ran_output_port_info = check_init
            self.ran_set_input_port_type = check_init
            self.ran_set_output_port_type = check_init
            self.ran_available_config = check_init
            self.ran_conf_info = check_init

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

        def _reconfigure(self, conf):
            self.ran_reconfigure = True

            self._base_reconfigure(conf)

        def _properties(self):
            self.ran_properties = True

            return self._base_properties()

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

        def __del__(self):
            if not self.ran_configure:
                test_error("_configure override was not called")
            if not self.ran_init:
                test_error("_init override was not called")
            if not self.ran_reset:
                test_error("_reset override was not called")
            # TODO: See TODO below.
            #if not self.ran_step:
            #    test_error("_step override was not called")
            #if not self.ran_reconfigure:
            #    test_error("_reconfigure override was not called")
            if not self.ran_properties:
                test_error("_properties override was not called")
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
    from sprokit.pipeline import process

    class PythonBaseExample(process.PythonProcess):
        def __init__(self, conf):
            process.PythonProcess.__init__(self, conf)

        def __del__(self):
            pass

    return PythonBaseExample


def base_example_process_cluster():
    from sprokit.pipeline import process
    from sprokit.pipeline import process_cluster

    class PythonBaseClusterExample(process_cluster.PythonProcessCluster):
        def __init__(self, conf):
            process_cluster.PythonProcessCluster.__init__(self, conf)

        def __del__(self):
            pass

    return PythonBaseClusterExample


def test_register():
    from sprokit.pipeline import config
    from sprokit.pipeline import process
    from sprokit.pipeline import process_factory

    proc_type = 'python_example'
    proc_desc = 'simple description'

    process_factory.add_process(proc_type, proc_desc, example_process(True))

    if not proc_desc == process_factory.description(proc_type):
        test_error("Description was not preserved when registering")

    try:
        p = process_factory.create_process(proc_type, process.ProcessName())
        if p is None:
            raise Exception()
    except:
        test_error("Could not create newly registered process type")


def test_register_cluster():
    from sprokit.pipeline import config
    from sprokit.pipeline import process
    from sprokit.pipeline import process_cluster
    from sprokit.pipeline import process_factory

    proc_type = 'python_example'
    proc_desc = 'simple description'

    process_factory.add_process(proc_type, proc_desc, base_example_process_cluster())

    if not proc_desc == process_factory.description(proc_type):
        test_error("Description was not preserved when registering")

    p = None

    try:
        p = process_factory.create_process(proc_type, process.ProcessName())
        if p is None:
            raise Exception()
    except BaseException:
        import sys

        e = sys.exc_info()[1]

        test_error("Could not create newly registered process cluster type: %s" % str(e))

    if process_cluster.cluster_from_process(p) is None:
        test_error("A cluster process from the registry was not detected as a cluster process")


def test_wrapper_api():
    from sprokit.pipeline import config
    from sprokit.pipeline import edge
    from sprokit.pipeline import process
    from sprokit.pipeline import process_factory

    proc_type = 'python_example'
    proc_desc = 'simple description'

    proc_base_type = 'python_base_example'
    proc_base_desc = 'simple base description'

    iport = 'no_such_iport'
    oport = 'no_such_oport'
    key = 'no_such_key'
    ptype = 'no_type'

    process_factory.add_process(proc_type, proc_desc, example_process(False))
    process_factory.add_process(proc_base_type, proc_base_desc, base_example_process())

    def check_process(p):
        if p is None:
            test_error("Got a 'None' process")
            return

        p.properties()

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
        p.available_tunable_config()
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

        # TODO: Can't check this because only the process_cluster base class
        # and the pipeline may reconfigure a process. Needs to be stepped
        # within a pipeline to verify this. Enable the ran_step check in
        # p.check when this is fixed.
        #p.reconfigure(reconf)

        del p

    p = process_factory.create_process(proc_type, process.ProcessName())
    check_process(p)

    p = process_factory.create_process(proc_base_type, process.ProcessName())
    check_process(p)


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
