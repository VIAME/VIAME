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

    vistk.pipeline.schedule_registry:default_type

    vistk.pipeline.load_known_modules()

    local reg = vistk.pipeline.schedule_registry.self()

    local sched_type = 'thread_per_process'
    local c = vistk.pipeline.empty_config()
    local p = vistk.pipeline.pipeline(c)

    reg:create_schedule(sched_type, c, p)
    reg:types()
    reg:description(sched_type)
end


function schedule_example(conf, pipe)
    -- TODO: How to do this?
    return nil
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

    reg:register_schedule(sched_type, sched_desc, schedule_example)

    if sched_desc ~= reg:description(sched_type) then
        log("Error: Description was not preserved when registering")
    end

    reg:create_schedule(sched_type, c, p)

    if not pcall(reg.create_schedule, sched_type, c, p) then
        log("Error: Could not create newly registered schedule type")
    end
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
