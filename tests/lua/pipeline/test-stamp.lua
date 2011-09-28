#!/usr/bin/env python
--ckwg +5
-- Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
-- KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
-- Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.


function log(msg)
    io.stderr:write(string.format("%s\n", msg))
end


function test_import()
    if not pcall(require, "vistk.pipeline.stamp") then
        log("Error: Failed to import the stamp module")
    end
end


function test_create()
    require("vistk.pipeline.stamp")

    vistk.pipeline.new_stamp()
end


function test_api_calls()
    require("vistk.pipeline.stamp")

    local s = vistk.pipeline.new_stamp()
    local sc = vistk.pipeline.copied_stamp(s)
    local si = vistk.pipeline.incremented_stamp(s)
    local t = vistk.pipeline.new_stamp()
    local sr = vistk.pipeline.recolored_stamp(s, t)

    if s:is_same_color(t) then
        log("Error: New stamps have the same color")
    end

    if not s:is_same_color(sc) then
        log("Error: Copied stamps do not have the same color")
    end

    if s > si then
        log("Error: A stamp is greater than its increment")
    end

    if si < s then
        log("Error: A stamp is greater than its increment")
    end

    if s < t or t < s then
        log("Error: Different colored stamps return True for comparison")
    end
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
