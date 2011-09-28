#!/usr/bin/env python
--ckwg +5
-- Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
-- KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
-- Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


function log(msg)
    io.stderr:write(string.format("%s\n", msg))
end


function test_import()
    if not pcall(require, "vistk.pipeline.pipeline") then
        log("Error: Failed to import the pipeline module")
    end
end


function test_create()
    require("vistk.pipeline.config")
    require("vistk.pipeline.pipeline")

    local c = vistk.pipeline.empty_config()

    vistk.pipeline.pipeline(c)
end


function test_api_calls()
    require("vistk.pipeline.config")
    require("vistk.pipeline.edge")
    require("vistk.pipeline.modules")
    require("vistk.pipeline.pipeline")
    require("vistk.pipeline.process")
    require("vistk.pipeline.process_registry")

    local c = vistk.pipeline.empty_config()

    local p = vistk.pipeline.pipeline(c)

    local proc_type1 = 'numbers'
    local proc_type2 = 'print_number'

    local proc_name1 = 'src'
    local proc_name2 = 'snk'

    local port_name1 = 'number'
    local port_name2 = 'number'

    local group_name = 'group'
    local group_iport = 'iport'
    local group_oport = 'oport'

    vistk.pipeline.load_known_modules()

    local reg = vistk.pipeline.process_registry.self()

    c:set_value('_name', proc_name1)
    --c:set_value(vistk.pipeline.process.config_name, proc_name1)
    local proc1 = reg:create_process(proc_type1, c)

    local conf_name = 'output'

    c:set_value('_name', proc_name2)
    --c:set_value(vistk.pipeline.process.config_name, proc_name2)
    c:set_value(conf_name, 'test-python-pipeline-api_calls-print_number.txt')
    local proc2 = reg:create_process(proc_type2, c)

    p:add_process(proc1)
    p:add_process(proc2)
    p:add_group(group_name)
    p:connect(proc_name1, port_name1,
              proc_name2, port_name2)
    p:map_input_port(group_name, group_iport,
                     proc_name2, port_name2,
                     vistk.pipeline.port_flags())
    p:map_output_port(group_name, group_oport,
                      proc_name1, port_name1,
                      vistk.pipeline.port_flags())
    p:process_names()
    p:process_by_name(proc_name1)
    p:upstream_for_process(proc_name2)
    p:upstream_for_port(proc_name2, port_name2)
    p:downstream_for_process(proc_name1)
    p:downstream_for_port(proc_name1, port_name1)
    p:sender_for_port(proc_name2, port_name2)
    p:receivers_for_port(proc_name1, port_name1)
    p:edge_for_connection(proc_name1, port_name1,
                          proc_name2, port_name2)
    p:input_edges_for_process(proc_name2)
    p:input_edge_for_port(proc_name2, port_name2)
    p:output_edges_for_process(proc_name1)
    p:output_edges_for_port(proc_name1, port_name1)
    p:groups()
    p:input_ports_for_group(group_name)
    p:output_ports_for_group(group_name)
    p:mapped_group_input_port_flags(group_name, group_iport)
    p:mapped_group_output_port_flags(group_name, group_oport)
    p:mapped_group_input_ports(group_name, group_iport)
    p:mapped_group_output_port(group_name, group_oport)

    p:setup_pipeline()
end


function main(testname)
    if testname == 'import' then
        test_import()
    elseif testname == 'create' then
        test_create()
    elseif testname == 'api_calls' then
        test_api_calls()
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
