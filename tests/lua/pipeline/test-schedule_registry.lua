#!/usr/bin/env python
--ckwg +5
-- Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
-- KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
-- Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


function log(msg)
    io.stderr:write(string.format("%s\n", msg))
end


function test_import()
    if not pcall(require, "vistk.pipeline.schedule_registry") then
        log("Error: Failed to import the schedule_registry module")
    end
end


function test_create()
    require("vistk.pipeline.schedule_registry")

    vistk.pipeline.schedule_registry.self()
    vistk.pipeline.schedule_type()
    vistk.pipeline.schedule_types()
    vistk.pipeline.schedule_description()
end


function test_api_calls()
    require("vistk.pipeline.config")
    require("vistk.pipeline.modules")
    require("vistk.pipeline.pipeline")
    require("vistk.pipeline.schedule_registry")

    local tmp

    tmp = vistk.pipeline.schedule_registry.default_type

    vistk.pipeline.load_known_modules()

    local reg = vistk.pipeline.schedule_registry.self()

    local sched_type = 'thread_per_process'
    local c = vistk.pipeline.empty_config()
    local p = vistk.pipeline.pipeline(c)

    reg:create_schedule(sched_type, c, p)
    reg:types()
    reg:description(sched_type)
end


function example_schedule(conf, pipe)
    require("vistk.pipeline.schedule")

    class 'lua_example' (vistk.pipeline.lua_schedule)

    function lua_example:__init(conf, pipe)
        vistk.pipeline.lua_schedule:__init(conf, pipe)

        self.ran_start = false
        self.ran_wait = false
        self.ran_stop = false
    end

    function lua_example:start()
        self.ran_start = true
    end

    function lua_example:wait()
        self.ran_wait = true
    end

    function lua_example:stop()
        self.ran_stop = true
    end

    function lua_example:check()
        if not self.ran_start then
            log("Error: start override was not called")
        end
        if not self.ran_wait then
            log("Error: wait override was not called")
        end
        if not self.ran_stop then
            log("Error: stop override was not called")
        end
    end

    function mk_example(conf, pipe)
        return lua_example(conf, pipe)
    end

    return mk_example
end


function test_register()
    require("vistk.pipeline.config")
    require("vistk.pipeline.modules")
    require("vistk.pipeline.pipeline")
    require("vistk.pipeline.schedule")
    require("vistk.pipeline.schedule_registry")

    vistk.pipeline.load_known_modules()

    local reg = vistk.pipeline.schedule_registry.self()

    local sched_type = 'python_example'
    local sched_desc = 'simple description'
    local c = vistk.pipeline.empty_config()
    local p = vistk.pipeline.pipeline(c)

    reg:register_schedule(sched_type, sched_desc, example_schedule())

    if sched_desc ~= reg:description(sched_type) then
        log("Error: Description was not preserved when registering")
    end

    reg:create_schedule(sched_type, c, p)

    if not pcall(reg.create_schedule, sched_type, c, p) then
        log("Error: Could not create newly registered schedule type")
    end
end


function test_wrapper_api()
    require("vistk.pipeline.config")
    require("vistk.pipeline.modules")
    require("vistk.pipeline.pipeline")
    require("vistk.pipeline.schedule_registry")

    local sched_type = 'lua_example'
    local sched_desc = 'simple description'

    local reg = vistk.pipeline.schedule_registry.self()

    reg:register_schedule(sched_type, sched_desc, example_schedule)

    local c = vistk.pipeline.empty_config()
    local p = vistk.pipeline.pipeline(c)

    function check_schedule(s)
        s:start()
        s:wait()
        s:stop()

        s:check()
    end

    local s = reg:create_schedule(sched_type, c, p)
    check_schedule(s)
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
