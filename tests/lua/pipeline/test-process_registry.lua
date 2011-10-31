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
    require("vistk.pipeline.process")

    class 'lua_example' (vistk.pipeline.lua_process)

    function lua_example:__init(conf)
        vistk.pipeline.lua_process:__init(conf)

        self.ran_init = false
        self.ran_step = false
        self.ran_connect_input_port = false
        self.ran_connect_output_port = false
        self.ran_input_ports = false
        self.ran_output_ports = false
        self.ran_input_port_info = false
        self.ran_output_port_info = false
        self.ran_available_config = false
        self.ran_conf_info = false
    end

    function lua_example:_init()
        self.ran_init = true

        vistk.pipeline.lua_process:_init()
    end

    function lua_example:_step()
        self.ran_step = true

        self.check()

        vistk.pipeline.lua_process:_step()
    end

    function lua_example:_connect_input_port(self, port, edge)
        self.ran_connect_input_port = true

        vistk.pipeline.lua_process:_connect_input_port(port, edge)
    end

    function lua_example:_connect_output_port(self, port, edge)
        self.ran_connect_output_port = true

        vistk.pipeline.lua_process:_connect_output_port(port, edge)
    end

    function lua_example:_input_ports()
        self.ran_input_ports = true

        return vistk.pipeline.lua_process:_input_ports()
    end

    function lua_example:_output_ports()
        self.ran_output_ports = true

        return vistk.pipeline.lua_process:_output_ports()
    end

    function lua_example:_input_port_info(self, port)
        self.ran_input_port_info = true

        return vistk.pipeline.lua_process:_input_port_info(port)
    end

    function lua_example:_output_port_info(self, port)
        self.ran_output_port_info = true

        return vistk.pipeline.lua_process:_output_port_info(port)
    end

    function lua_example:_available_config()
        self.ran_available_config = true

        return vistk.pipeline.lua_process:_available_config()
    end

    function lua_example:_config_info(self, key)
        self.ran_conf_info = true

        return vistk.pipeline.lua_process:_conf_info(key)
    end

    function lua_example:check()
        if not self.ran_init then
            log("Error: _init override was not called")
        end
        if not self.ran_step then
            log("Error: _step override was not called")
        end
        if not self.ran_connect_input_port then
            log("Error: _connect_input_port override was not called")
        end
        if not self.ran_connect_output_port then
            log("Error: _connect_output_port override was not called")
        end
        if not self.ran_input_ports then
            log("Error: _input_ports override was not called")
        end
        if not self.ran_output_ports then
            log("Error: _output_ports override was not called")
        end
        if not self.ran_input_port_info then
            log("Error: _input_port_info override was not called")
        end
        if not self.ran_output_port_info then
            log("Error: _output_port_info override was not called")
        end
        if not self.ran_available_config then
            log("Error: _available_config override was not called")
        end
        if not self.ran_conf_info then
            log("Error: _conf_info override was not called")
        end
    end

    function mk_example(conf)
        return lua_example(conf)
    end

    return mk_example
end


function base_example_process()
    require("vistk.pipeline.process")

    class 'lua_base_example' (vistk.pipeline.lua_process)

    function lua_base_example:__init(conf)
        vistk.pipeline.lua_process:__init(conf)
    end

    function lua_base_example:check()
    end

    function mk_base_example(conf)
        return lua_base_example(conf)
    end

    return mk_base_example
end


function test_register()
    require("vistk.pipeline.config")
    require("vistk.pipeline.process")
    require("vistk.pipeline.process_registry")

    local proc_type = 'python_example'
    local proc_desc = 'simple description'

    local reg = vistk.pipeline.process_registry.self()

    reg:register_process(proc_type, proc_desc, example_process())

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

    reg:register_process(proc_type, proc_desc, example_process())
    reg:register_process(proc_base_type, proc_base_desc, base_example_process())

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

        p:check()
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
