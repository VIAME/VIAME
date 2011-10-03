#!/usr/bin/env python
--ckwg +5
-- Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
-- KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
-- Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


function log(msg)
    io.stderr:write(string.format("%s\n", msg))
end


function test_import()
    if not pcall(require, "vistk.pipeline_util.load") then
        log("Error: Failed to import the load module")
    end
end


function test_create()
    require("vistk.pipeline_util.load")

    vistk.pipeline_util.token()
    vistk.pipeline_util.config_flag()
    vistk.pipeline_util.config_flags()
    vistk.pipeline_util.config_provider()
    vistk.pipeline_util.config_key_options()
    vistk.pipeline_util.config_key()
    vistk.pipeline_util.config_value()
    vistk.pipeline_util.config_values()
    vistk.pipeline_util.map_options()
    vistk.pipeline_util.input_map()
    vistk.pipeline_util.input_maps()
    vistk.pipeline_util.output_map()
    vistk.pipeline_util.output_maps()
    vistk.pipeline_util.config_block()
    vistk.pipeline_util.process_block()
    vistk.pipeline_util.connect_block()
    vistk.pipeline_util.group_block()
    vistk.pipeline_util.pipe_block()
    vistk.pipeline_util.pipe_blocks()
end


function test_api_calls()
    require("vistk.pipeline.config")
    require("vistk.pipeline.process")
    require("vistk.pipeline.process_registry")
    require("vistk.pipeline_util.load")

    local tmp
    local o

    o = vistk.pipeline_util.config_key_options()
    tmp = o.flags
    tmp = o.provider
    o.flags = vistk.pipeline_util.config_flags()
    o.provider = vistk.pipeline_util.config_provider()

    o = vistk.pipeline_util.config_key()
    tmp = o.key_path
    tmp = o.options
    o.key_path = vistk.pipeline.config_keys()
    o.options = vistk.pipeline_util.config_key_options()

    o = vistk.pipeline_util.config_value()
    tmp = o.key
    tmp = o.value
    o.key = vistk.pipeline_util.config_key()
    o.value = vistk.pipeline.config_value()

    o = vistk.pipeline_util.map_options()
    tmp = o.flags
    o.flags = vistk.pipeline.port_flags()

    o = vistk.pipeline_util.input_map()
    tmp = o.options
    tmp = o.from
    tmp = o.to
    o.options = vistk.pipeline_util.map_options()
    o.from = vistk.pipeline.port()
    o.to = vistk.pipeline.port_addr()

    o = vistk.pipeline_util.output_map()
    tmp = o.options
    tmp = o.from
    tmp = o.to
    o.options = vistk.pipeline_util.map_options()
    o.from = vistk.pipeline.port_addr()
    o.to = vistk.pipeline.port()

    o = vistk.pipeline_util.config_block()
    tmp = o.key
    tmp = o.values
    o.key = vistk.pipeline.config_keys()
    o.values = vistk.pipeline_util.config_values()

    o = vistk.pipeline_util.process_block()
    tmp = o.name
    tmp = o.type
    tmp = o.config_values
    o.name = vistk.pipeline.process_name()
    o.type = process_registry.process_type()
    o.config_values = vistk.pipeline_util.config_values()

    o = vistk.pipeline_util.connect_block()
    tmp = o.from
    tmp = o.to
    o.from = vistk.pipeline.port_addr()
    o.to = vistk.pipeline.port_addr()

    o = vistk.pipeline_util.group_block()
    tmp = o.name
    tmp = o.config_values
    tmp = o.input_mappings
    tmp = o.output_mappings
    o.name = vistk.pipeline.process_name()
    o.config_values = vistk.pipeline_util.config_values()
    o.input_mappings = vistk.pipeline_util.input_maps()
    o.output_mappings = vistk.pipeline_util.output_maps()

    o = vistk.pipeline_util.pipe_block()
    o.config = vistk.pipeline_util.config_block()
    tmp = o.config
    o.process = vistk.pipeline_util.process_block()
    tmp = o.process
    o.connect = vistk.pipeline_util.connect_block()
    tmp = o.connect
    o.group = vistk.pipeline_util.group_block()
    tmp = o.group
end


function test_simple_pipeline(path)
    require("vistk.pipeline_util.load")

    local blocks = vistk.pipeline_util.load_pipe_file(path)
    local fin = io.open(path, 'r')
    vistk.pipeline_util.load_pipe(fin)
    fin:close()
end


function main(testname, path)
    if testname == 'import' then
        test_import()
    elseif testname == 'create' then
        test_create()
    elseif testname == 'api_calls' then
        test_api_calls()
    elseif testname == 'simple_pipeline' then
        test_simple_pipeline(path)
    else
        log(string.format("Error: No such test '%s'", testname))
    end
end


if #arg ~= 3 then
    log("Error: Expected three arguments")
    os.exit(1)
end

local testname = arg[1]

package.cpath = string.format("%s/?@CMAKE_SHARED_LIBRARY_SUFFIX@;%s", arg[2], package.cpath)

local pipeline_dir = arg[3]

local path = string.format("%s/%s.pipe", pipeline_dir, testname)

local status, err = pcall(main, testname, path)

if not status then
    log(string.format("Error: Unexpected exception: %s", err))
end
