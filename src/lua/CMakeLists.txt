project(vistk_lua)

find_package(Lua51 REQUIRED)
include_directories(SYSTEM ${LUA_INCLUDE_DIR})

find_package(Luabind REQUIRED)
include_directories(SYSTEM ${LUABIND_INCLUDE_DIR})

add_custom_target(lua)

function (add_vistk_lua_library name modpath)
  add_library(lua-${name}
    ${ARGN})

  if (WIN32)
    set(module_output RUNTIME_OUTPUT_DIRECTORY)
  else (WIN32)
    set(module_output LIBRARY_OUTPUT_DIRECTORY)
  endif (WIN32)

  set_target_properties(lua-${name}
    PROPERTIES
      OUTPUT_NAME      ${name}
      ${module_output} "${LUA_OUTPUT_PATH}/${modpath}"
      PREFIX           "")

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    if (WIN32)
      set(output_path "${LUA_OUTPUT_PATH}/${config}")
    endif (WIN32)

    string(TOUPPER "${config}" upper_config)

    set_target_properties(lua-${name}
      PROPERTIES
        "${module_output}_${upper_config}" "${output_path}/${modpath}")
  endforeach (config)

  install(
    TARGETS     lua-${name}
    DESTINATION "${LUA_INSTALL_PATH}/${modpath}"
    COMPONENT   runtime)

  add_dependencies(lua
    lua-${name})
endfunction (add_vistk_lua_library name modpath)

add_subdirectory(pipeline)
