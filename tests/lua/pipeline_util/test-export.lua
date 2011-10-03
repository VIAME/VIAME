#!/usr/bin/env python
--ckwg +5
-- Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
-- KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
-- Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


function log(msg)
    io.stderr:write(string.format("%s\n", msg))
end


function test_import()
    if not pcall(require, "vistk.pipeline_util.export") then
        log("Error: Failed to import the export module")
    end
end


function test_simple_pipeline(path)
    require("ex")

    require("vistk.pipeline.pipeline")
    require("vistk.pipeline.modules")
    require("vistk.pipeline_util.bake")
    require("vistk.pipeline_util.export_")

    vistk.pipeline.load_known_modules()

    local p = vistk.pipeline_util.bake_pipe_file(path)
    local r, w = io.pipe()

    local name = 'graph'

    vistk.pipeline_util.export_dot(w, p, name)

    r:close()
    w:close()
end


function main(testname, path)
    if testname == 'import' then
        test_import()
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
