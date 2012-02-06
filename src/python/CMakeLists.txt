project(vistk_python)

find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    iostreams
    python)

find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_PATH})

add_subdirectory(helpers)

add_custom_target(python)

function (add_vistk_python_library name modpath)
  string(REPLACE "/" "." safe_modpath ${modpath})

  add_library(python-${safe_modpath}-${name}
    ${ARGN})

  if (WIN32)
    set(module_output RUNTIME_OUTPUT_DIRECTORY)
  else (WIN32)
    set(module_output LIBRARY_OUTPUT_DIRECTORY)
  endif (WIN32)

  set_target_properties(python-${safe_modpath}-${name}
    PROPERTIES
      OUTPUT_NAME      ${name}
      ${module_output} "${PYTHON_OUTPUT_PATH}/${modpath}"
      PREFIX           "")

  if (WIN32)
    set_target_properties(python-${safe_modpath}-${name}
      PROPERTIES
        SUFFIX ".pyd")
  endif (WIN32)

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    if (WIN32)
      set(output_path "${PYTHON_OUTPUT_PATH}/${config}")
    endif (WIN32)

    string(TOUPPER "${config}" upper_config)

    set_target_properties(python-${safe_modpath}-${name}
      PROPERTIES
        "${module_output}_${upper_config}" "${output_path}/${modpath}")
  endforeach (config)

  install(
    TARGETS     python-${safe_modpath}-${name}
    DESTINATION "${PYTHON_INSTALL_PATH}/${modpath}"
    COMPONENT   runtime)

  add_dependencies(python
    python-${safe_modpath}-${name})
endfunction (add_vistk_python_library name modpath)

function (make_python_init_file modpath)
  if (WIN32)
    set(module_output RUNTIME_OUTPUT_DIRECTORY)
  else (WIN32)
    set(module_output LIBRARY_OUTPUT_DIRECTORY)
  endif (WIN32)

  set(import_lines)

  foreach (module ${ARGN})
    set(import_lines
      "${import_lines}\nfrom ${module} import *")
  endforeach (module)

  set(copyright_header
"# ckwg +5
# Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.")

  file(WRITE "${PYTHON_OUTPUT_PATH}/${modpath}/__init__.py"
"${copyright_header}
${import_lines}
")

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    if (WIN32)
      set(output_path "${PYTHON_OUTPUT_PATH}/${config}")
    endif (WIN32)

    file(WRITE "${output_path}/${modpath}/__init__.py"
"${copyright_header}
${import_lines}
")
  endforeach (config)

  if (NOT ${no_install_plugin})
    install(
      FILES       "${PYTHON_OUTPUT_PATH}/${modpath}/__init__.py"
      DESTINATION "${PYTHON_INSTALL_PATH}/${modpath}"
      COMPONENT   runtime)
  endif (NOT ${no_install_plugin})

  add_dependencies(python
    python-init-${name})
endfunction (make_python_init_file modpath)

add_subdirectory(image)
add_subdirectory(pipeline)
add_subdirectory(pipeline_util)

if (VISTK_ENABLE_TESTING)
  add_subdirectory(test)
endif (VISTK_ENABLE_TESTING)

make_python_init_file(vistk)
#  image
#  pipeline
#  pipeline_util)
