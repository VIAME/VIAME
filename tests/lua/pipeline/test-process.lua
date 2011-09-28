#!/usr/bin/env python
--ckwg +5
-- Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
-- KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
-- Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


function log(msg)
    io.stderr:write(string.format("%s\n", msg))
end


function test_import()
    if not pcall(require, "vistk.pipeline.process") then
        log("Error: Failed to import the process module")
    end
end


function test_create()
    require("vistk.pipeline.datum")
    require("vistk.pipeline.process")

    vistk.pipeline.edge_ref()
    vistk.pipeline.edge_group()
    vistk.pipeline.process_name()
    vistk.pipeline.process_names()
    vistk.pipeline.port_description()
    vistk.pipeline.port()
    vistk.pipeline.ports()
    vistk.pipeline.port_type()
    vistk.pipeline.port_flag()
    vistk.pipeline.port_flags()
    vistk.pipeline.port_addr()
    vistk.pipeline.port_addrs()
    vistk.pipeline.port_info('type', vistk.pipeline.port_flags(), 'desc')
    vistk.pipeline.conf_info('default', 'desc')
    vistk.pipeline.data_info(true, true, vistk.pipeline.datum.invalid)
end


function test_api_calls()
    require("vistk.pipeline.datum")
    require("vistk.pipeline.process")

    local tmp

    a = vistk.pipeline.port_addr()
    tmp = a.process
    tmp = a.port
    a.process = ''
    a.port = ''

    a = vistk.pipeline.port_info('type', vistk.pipeline.port_flags(), 'desc')
    tmp = a.type
    tmp = a.flags
    tmp = a.description

    a = vistk.pipeline.conf_info('default', 'desc')
    tmp = a.default
    tmp = a.description

    a = vistk.pipeline.data_info(true, true, vistk.pipeline.datum.invalid)
    tmp = a.same_color
    tmp = a.in_sync
    tmp = a.max_status
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
