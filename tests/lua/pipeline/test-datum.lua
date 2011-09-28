#!/usr/bin/env lua
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
    if not pcall(require, "vistk.pipeline.datum") then
        log("Error: Failed to import the datum module")
    end
end


function test_empty()
    require("vistk.pipeline.datum")

    local d = vistk.pipeline.empty_datum()

    if d:type() ~= vistk.pipeline.datum.empty then
        log("Error: Datum type mismatch")
    end

    if string.len(d:get_error()) ~= 0 then
        log("Error: A empty datum has an error string")
    end

--    ensure_exception("retrieving data from an empty datum",
--                     d.get_datum)
end


function test_complete()
    require("vistk.pipeline.datum")

    local d = vistk.pipeline.complete_datum()

    if d:type() ~= vistk.pipeline.datum.complete then
        log("Error: Datum type mismatch")
    end

    if string.len(d:get_error()) ~= 0 then
        log("Error: A complete datum has an error string")
    end

--    ensure_exception("retrieving data from a complete datum",
--                     d.get_datum)
end


function test_error()
    require("vistk.pipeline.datum")

    local err = 'An error'

    local d = vistk.pipeline.error_datum(err)

    if d:type() ~= vistk.pipeline.datum.error then
        log("Error: Datum type mismatch")
    end

    if d:get_error() ~= err then
        log("Error: An error datum did not keep the message")
    end

--    ensure_exception("retrieving data from an error datum",
--                     d.get_datum)
end


function main(testname)
    if testname == 'import' then
        test_import()
    elseif testname == 'empty' then
        test_empty()
    elseif testname == 'complete' then
        test_complete()
    elseif testname == 'error' then
        test_error()
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
