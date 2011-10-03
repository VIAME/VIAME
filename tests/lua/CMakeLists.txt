project(vistk_test_lua)

find_package(LuaInterp REQUIRED)

set(TEST_WORKING_DIRECTORY
  ${EXECUTABLE_OUTPUT_PATH})
set(TEST_RUNNER
  ${LUA_EXECUTABLE})

function (vistk_build_lua_test group input)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${input}
    ${EXECUTABLE_OUTPUT_PATH}/test-lua-${group})

  vistk_declare_test(lua-${group})
endfunction (vistk_build_lua_test)

function (vistk_make_lua_test group instance)
  set(lua_module_path
    ${LUA_OUTPUT_PATH})

  if (WIN32)
    set(lua_module_path
      "${lua_module_path}/$<CONFIGURATION>")
  endif (WIN32)

  vistk_make_test(lua-${group} ${instance}
    ${lua_module_path} ${ARGN})
endfunction (vistk_make_lua_test)

add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
