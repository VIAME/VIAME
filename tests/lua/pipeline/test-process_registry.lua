#!/usr/bin/env python
--ckwg +5
-- Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
-- KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
-- Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


function log(msg)
    io.stderr:write(string.format("%s\n", msg))
end


function ensure_exception(action, func, ...)
    local got_exception = false

    if not pcall(func, ...) then
        got_exception = true
    end

    if not got_exception then
        log(string.format("Error: Did not get exception when %s", action))
    end
end


function test_import()
    if not pcall(require, "vistk.pipeline.process_registry") then
        log("Error: Failed to import the process_registry module")
    end
end


function test_create()
    require("vistk.pipeline.process_registry")

    vistk.pipeline.process_registry.self()
    vistk.pipeline.process_type()
    vistk.pipeline.process_types()
    vistk.pipeline.process_description()
end


function test_api_calls()
    require("vistk.pipeline.config")
    require("vistk.pipeline.modules")
    require("vistk.pipeline.process_registry")

    vistk.pipeline.load_known_modules()

    local reg = vistk.pipeline.process_registry.self()

    local proc_type = 'orphan'
    local c = vistk.pipeline.empty_config()

    reg:create_process(proc_type, c)
    reg:types()
    reg:description(proc_type)
end


function example_process()
    -- TODO: How to do this?
    return nil
--    require("vistk.pipeline.process")
--
--    class PythonExample(process.PythonProcess):
--        function __init__(self, conf)
--            process.PythonProcess.__init__(self, conf)
--
--            self.ran_init = False
--            self.ran_step = False
--            self.ran_connect_input_port = False
--            self.ran_connect_output_port = False
--            self.ran_input_ports = False
--            self.ran_output_ports = False
--            self.ran_input_port_info = False
--            self.ran_output_port_info = False
--            self.ran_available_config = False
--            self.ran_conf_info = False
--        end
--
--        function _init(self)
--            self.ran_init = True
--
--            self._base_init()
--        end
--
--        function _step(self)
--            self.ran_step = True
--
--            self.check()
--
--            self._base_step()
--        end
--
--        function _connect_input_port(self, port, edge)
--            self.ran_connect_input_port = True
--
--            self._base_connect_input_port(port, edge)
--        end
--
--        function _connect_output_port(self, port, edge)
--            self.ran_connect_output_port = True
--
--            self._base_connect_output_port(port, edge)
--        end
--
--        function _input_ports(self)
--            self.ran_input_ports = True
--
--            return self._base_input_ports()
--        end
--
--        function _output_ports(self)
--            self.ran_output_ports = True
--
--            return self._base_output_ports()
--        end
--
--        function _input_port_info(self, port)
--            self.ran_input_port_info = True
--
--            return self._base_input_port_info(port)
--        end
--
--        function _output_port_info(self, port)
--            self.ran_output_port_info = True
--
--            return self._base_output_port_info(port)
--        end
--
--        function _available_config(self)
--            self.ran_available_config = True
--
--            return self._base_available_config()
--        end
--
--        function _config_info(self, key)
--            self.ran_conf_info = True
--
--            return self._base_conf_info(key)
--        end
--
--        function check(self)
--            if not self.ran_init then
--                log("Error: _init override was not called")
--            end
--            if not self.ran_step then
--                log("Error: _step override was not called")
--            end
--            if not self.ran_connect_input_port then
--                log("Error: _connect_input_port override was not called")
--            end
--            if not self.ran_connect_output_port then
--                log("Error: _connect_output_port override was not called")
--            end
--            if not self.ran_input_ports then
--                log("Error: _input_ports override was not called")
--            end
--            if not self.ran_output_ports then
--                log("Error: _output_ports override was not called")
--            end
--            if not self.ran_input_port_info then
--                log("Error: _input_port_info override was not called")
--            end
--            if not self.ran_output_port_info then
--                log("Error: _output_port_info override was not called")
--            end
--            if not self.ran_available_config then
--                log("Error: _available_config override was not called")
--            end
--            if not self.ran_conf_info then
--                log("Error: _conf_info override was not called")
--            end
--        end
end


function base_example_process()
    -- TODO: How to do this?
    return nil
--    require("vistk.pipeline.process")
--
--    class PythonBaseExample(process.PythonProcess):
--        function __init__(self, conf)
--            process.PythonProcess.__init__(self, conf)
--        end
--
--    return PythonBaseExample
end


function test_register()
    require("vistk.pipeline.config")
    require("vistk.pipeline.process")
    require("vistk.pipeline.process_registry")

    local proc_type = 'python_example'
    local proc_desc = 'simple description'

    local reg = vistk.pipeline.process_registry.self()

    reg:register_process(proc_type, proc_desc, example_process)

    if proc_desc ~= reg:description(proc_type) then
        log("Error: Description was not preserved when registering")
    end

    local c = vistk.pipeline.empty_config()

    reg:create_process(proc_type, c)

    if not pcall(reg.create_process, proc_type, c) then
        log("Error: Could not create newly registered process type")
    end
end


function test_wrapper_api()
    require("vistk.pipeline.config")
    require("vistk.pipeline.edge")
    require("vistk.pipeline.process")
    require("vistk.pipeline.process_registry")

    local proc_type = 'python_example'
    local proc_desc = 'simple description'

    local proc_base_type = 'python_base_example'
    local proc_base_desc = 'simple base description'

    local iport = 'no_such_iport'
    local oport = 'no_such_oport'
    local key = 'no_such_key'

    local reg = vistk.pipeline.process_registry.self()

    reg:register_process(proc_type, proc_desc, example_process)
    reg:register_process(proc_base_type, proc_base_desc, base_example_process)

    local c = vistk.pipeline.empty_config()

    function check_process(p)
        p:input_ports()
        p:output_ports()
        ensure_exception("asking for info on a non-existant input port",
                         p.input_port_info, iport)
        ensure_exception("asking for info on a non-existant output port",
                         p.output_port_info, oport)

        local e = vistk.pipeline.edge(c)

        ensure_exception("connecting to a non-existant input port",
                         p.connect_input_port, iport, e)
        ensure_exception("connecting to a non-existant output port",
                         p.connect_output_port, oport, e)

        p:available_config()
        ensure_exception("asking for info on a non-existant config key",
                         p.config_info, key)

        p:init()
        p:step()
    end

    local p
    p = reg:create_process(proc_type, c)
    check_process(p)

    p = reg:create_process(proc_base_type, c)
    check_process(p)
end


function main(testname)
    if testname == 'import' then
        test_import()
    elseif testname == 'create' then
        test_create()
    elseif testname == 'api_calls' then
        test_api_calls()
    elseif testname == 'register' then
        test_register()
    elseif testname == 'wrapper_api' then
        test_wrapper_api()
    else
        log(string.format("Error: No such test '%s'", testname))
    end
end


if #arg ~= 2 then
    log("Error: Expected two arguments")
    os.exit(1)
end

local testname = arg[1]

package.cpath = string.format("%s/?@CMAKE_SHARED_LIBRARY_SUFFIX@;%s", arg[2], package.cpath)

local status, err = pcall(main, testname)

if not status then
    log(string.format("Error: Unexpected exception: %s", err))
end
