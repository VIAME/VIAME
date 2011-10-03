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

function (make_lua_init_file modpath)
  if (WIN32)
    set(module_output RUNTIME_OUTPUT_DIRECTORY)
  else (WIN32)
    set(module_output LIBRARY_OUTPUT_DIRECTORY)
  endif (WIN32)

  set(import_lines)

  foreach (module ${ARGN})
    set(import_lines
      "${import_lines}\nrequire(\"${module}\")")
  endforeach (module)

  set(copyright_header
"# ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.")

file(WRITE "${LUA_OUTPUT_PATH}/${modpath}/init.lua"
"${copyright_header}
${import_lines}
")

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    if (WIN32)
      set(output_path "${LUA_OUTPUT_PATH}/${config}")
    endif (WIN32)

    file(WRITE "${output_path}/${modpath}/init.lua"
"${copyright_header}
${import_lines}
")
  endforeach (config)

  install(
    FILES       "${LUA_OUTPUT_PATH}/${modpath}/init.lua"
    DESTINATION "${LUA_INSTALL_PATH}/${modpath}"
    COMPONENT   runtime)

  add_dependencies(lua
    lua-init-${name})
endfunction (make_lua_init_file modpath)

add_subdirectory(pipeline)
