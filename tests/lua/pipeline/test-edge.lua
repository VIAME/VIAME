#!/usr/bin/env lua
--ckwg +5
-- Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
-- KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
-- Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


function log(msg)
    io.stderr:write(string.format("%s\n", msg))
end


function test_import()
    if not pcall(require, "vistk.pipeline.edge") then
        log("Error: Failed to import the edge module")
    end
end


function test_create()
    require("vistk.pipeline.edge")
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    vistk.pipeline.edge(c)
    vistk.pipeline.edges()
end


function test_datum_create()
    require("vistk.pipeline.datum")
    require("vistk.pipeline.edge")
    require("vistk.pipeline.stamp")

    local d = vistk.pipeline.complete_datum()
    local s = vistk.pipeline.new_stamp()

    vistk.pipeline.edge_datum(d, s)
    vistk.pipeline.edge_data()
end


function test_api_calls()
    require("vistk.pipeline.config")
    require("vistk.pipeline.datum")
    require("vistk.pipeline.edge")
    require("vistk.pipeline.modules")
    require("vistk.pipeline.process_registry")
    require("vistk.pipeline.stamp")

    local c = vistk.pipeline.empty_config()

    local e = vistk.pipeline.edge(c)

    e:makes_dependency()
    e:has_data()
    e:full_of_data()
    e:datum_count()

    local d = vistk.pipeline.complete_datum()
    local s = vistk.pipeline.new_stamp()

    local ed = vistk.pipeline.edge_datum(d, s)

    e:push_datum(ed)
    e:get_datum()

    e:push_datum(ed)
    e:peek_datum()
    e:pop_datum()

    vistk.pipeline.load_known_modules()

    local reg = vistk.pipeline.process_registry.self()

    local p = reg:create_process('orphan', c)

    e:set_upstream_process(p)
    e:set_downstream_process(p)

    e:mark_downstream_as_complete()
    e:is_downstream_complete()
end


function test_datum_api_calls()
    require("vistk.pipeline.datum")
    require("vistk.pipeline.edge")
    require("vistk.pipeline.stamp")

    local d = vistk.pipeline.complete_datum()
    local s = vistk.pipeline.new_stamp()

    local ed = vistk.pipeline.edge_datum(d, s)

    local tmp
    tmp = ed.datum
    ed.datum = d
    tmp = ed.stamp
    ed.stamp = s
end


function main(testname)
    if testname == 'import' then
        test_import()
    elseif testname == 'create' then
        test_create()
    elseif testname == 'datum_create' then
        test_datum_create()
    elseif testname == 'api_calls' then
        test_api_calls()
    elseif testname == 'datum_api_calls' then
        test_datum_api_calls()
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
