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
    if not pcall(require, "vistk.pipeline.config") then
        log("Error: Failed to import the config module")
    end
end


function test_create()
    require("vistk.pipeline.config")

    if not pcall(vistk.pipeline.empty_config) then
        log("Error: Failed to create an empty configuration")
    end

    vistk.pipeline.config_key()
    vistk.pipeline.config_keys()
    vistk.pipeline.config_value()
end


function test_has_value()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local keya = 'keya'
    local keyb = 'keyb'

    local valuea = 'value_a'

    c:set_value(keya, valuea)

    if not c:has_value(keya) then
        log("Error: Block does not have value which was set")
    end

    if c:has_value(keyb) then
        log("Error: Block has value which was not set")
    end
end


function test_get_value()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local keya = 'keya'

    local valuea = 'value_a'

    c:set_value(keya, valuea)

    local get_valuea = c:get_value(keya)

    if valuea ~= get_valuea then
        log("Error: Did not retrieve value that was set")
    end
end


function test_get_value_no_exist()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local keya = 'keya'
    local keyb = 'keyb'

    local valueb = 'value_b'

    ensure_exception("retrieving an unset value",
                     c.get_value, keya)

    local get_valueb = c:get_value(keyb, valueb)

    if valueb ~= get_valueb then
        log("Error: Did not retrieve default when requesting unset value")
    end
end


function test_unset_value()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local keya = 'keya'
    local keyb = 'keyb'

    local valuea = 'value_a'
    local valueb = 'value_b'

    c:set_value(keya, valuea)
    c:set_value(keyb, valueb)

    c:unset_value(keya)

    ensure_exception("retrieving an unset value",
                     c.get_value, keya)

    local get_valueb = c:get_value(keyb)

    if valueb ~= get_valueb then
        log("Error: Did not retrieve value when requesting after an unrelated unset")
    end
end


function test_available_values()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local keya = 'keya'
    local keyb = 'keyb'

    local valuea = 'value_a'
    local valueb = 'value_b'

    c:set_value(keya, valuea)
    c:set_value(keyb, valueb)

    local avail = c:available_values()

    if #avail ~= 2 then
        log("Error: Did not retrieve correct number of keys")
    end

    if not pcall(table.foreach, avail, function (id) end) then
        log("Error: Available values is not iterable")
    end
end


function test_read_only()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local keya = 'keya'

    local valuea = 'value_a'
    local valueb = 'value_b'

    c:set_value(keya, valuea)

    c:mark_read_only(keya)

    ensure_exception("setting a read only value",
                     c.set_value, keya, valueb)

    local get_valuea = c:get_value(keya)

    if valuea ~= get_valuea then
        log("Error: Read only value changed")
    end
end


function test_read_only_unset()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local keya = 'keya'

    local valuea = 'value_a'

    c:set_value(keya, valuea)

    c:mark_read_only(keya)

    ensure_exception("unsetting a read only value",
                     c.unset_value, keya)

    local get_valuea = c:get_value(keya)

    if valuea ~= get_valuea then
        log("Error: Read only value was unset")
    end
end


function test_subblock()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local block1 = 'block1'
    local block2 = 'block2'

    local keya = 'keya'
    local keyb = 'keyb'
    local keyc = 'keyc'

    local valuea = 'value_a'
    local valueb = 'value_b'
    local valuec = 'value_c'

    c:set_value(string.format('%s:%s', block1, keya), valuea)
    c:set_value(string.format('%s:%s', block1, keyb), valueb)
    c:set_value(string.format('%s:%s', block2, keyc), valuec)
    --c:set_value(string.format('%s%s%s', block1, vistk.pipeline.config.block_sep, keya), valuea)
    --c:set_value(string.format('%s%s%s', block1, vistk.pipeline.config.block_sep, keyb), valueb)
    --c:set_value(string.format('%s%s%s', block2, vistk.pipeline.config.block_sep, keyc), valuec)

    local d = c:subblock(block1)

    local get_valuea = d:get_value(keya)

    if valuea ~= get_valuea then
        log("Error: Subblock does not inherit expected keys")
    end

    local get_valueb = d:get_value(keyb)

    if valueb ~= get_valueb then
        log("Error: Subblock does not inherit expected keys")
    end

    if d:has_value(keyc) then
        log("Error: Subblock inherited unrelated key")
    end
end


function test_subblock_view()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()

    local block1 = 'block1'
    local block2 = 'block2'

    local keya = 'keya'
    local keyb = 'keyb'
    local keyc = 'keyc'

    local valuea = 'value_a'
    local valueb = 'value_b'
    local valuec = 'value_c'

    c:set_value(string.format('%s:%s', block1, keya), valuea)
    c:set_value(string.format('%s:%s', block2, keyb), valueb)
    --c:set_value(string.format('%s%s%s', block1, vistk.pipeline.config.block_sep, keya), valuea)
    --c:set_value(string.format('%s%s%s', block2, vistk.pipeline.config.block_sep, keyb), valueb)

    local d = c:subblock_view(block1)

    if not d:has_value(keya) then
        log("Error: Subblock does not inherit expected keys")
    end

    if d:has_value(keyb) then
        log("Error: Subblock inherited unrelated key")
    end

    c:set_value(string.format('%s:%s', block1, keya), valueb)
    --c:set_value(string.format('%s%s%s', block1, vistk.pipeline.config.block_sep, keya), valueb)

    local get_valuea1 = d:get_value(keya)

    if valueb ~= get_valuea1 then
        log("Error: Subblock view persisted a changed value")
    end

    d:set_value(keya, valuea)

    local get_valuea2 = d:get_value(keya)

    if valuea ~= get_valuea2 then
        log("Error: Subblock view set value was not changed in parent")
    end
end


function test_merge_config()
    require("vistk.pipeline.config")

    local c = vistk.pipeline.empty_config()
    local d = vistk.pipeline.empty_config()

    local keya = 'keya'
    local keyb = 'keyb'
    local keyc = 'keyc'

    local valuea = 'value_a'
    local valueb = 'value_b'
    local valuec = 'value_c'

    c:set_value(keya, valuea)
    c:set_value(keyb, valuea)

    d:set_value(keyb, valueb)
    d:set_value(keyc, valuec)

    c:merge_config(d)

    local get_valuea = c:get_value(keya)

    if valuea ~= get_valuea then
        log("Error: Unmerged key changed")
    end

    local get_valueb = c:get_value(keyb)

    if valueb ~= get_valueb then
        log("Error: Conflicting key was not overwritten")
    end

    local get_valuec = c:get_value(keyc)

    if valuec ~= get_valuec then
        log("Error: New key did not appear")
    end
end


function main(testname)
    if testname == 'import' then
        test_import()
    elseif testname == 'create' then
        test_create()
    elseif testname == 'has_value' then
        test_has_value()
    elseif testname == 'get_value' then
        test_get_value()
    elseif testname == 'get_value_no_exist' then
        test_get_value_no_exist()
    elseif testname == 'unset_value' then
        test_unset_value()
    elseif testname == 'available_values' then
        test_available_values()
    elseif testname == 'read_only' then
        test_read_only()
    elseif testname == 'read_only_unset' then
        test_read_only_unset()
    elseif testname == 'subblock' then
        test_subblock()
    elseif testname == 'subblock_view' then
        test_subblock_view()
    elseif testname == 'merge_config' then
        test_merge_config()
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
