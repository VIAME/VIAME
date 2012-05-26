project(vistk_python)

set(boost_python_library
  python)

if (VISTK_ENABLE_PYTHON3)
  set(boost_python_library
    python3)
endif (VISTK_ENABLE_PYTHON3)

string(TOUPPER ${boost_python_library} boost_python_library_upper)
set(boost_python_library_var
  Boost_${boost_python_library_upper}_LIBRARY)

find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    iostreams
    ${boost_python_library})

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

option(VISTK_ENABLE_BOOST_PYTHON_FIXES "Enables fixes for Boost.Python bugs (recommended)" ON)
mark_as_advanced(VISTK_ENABLE_BOOST_PYTHON_FIXES)

if (VISTK_ENABLE_BOOST_PYTHON_FIXES)
  # Not fixed yet.
  #if (${Boost_VERSION} LESS 10XX00)
    # Override Boost's invoke.hpp
    include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/helpers/include/threading")
  #endif (${Boost_VERSION} LESS 10XX00)
endif (VISTK_ENABLE_BOOST_PYTHON_FIXES)

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

  if (NOT ${no_install_plugin})
    install(
      TARGETS     python-${safe_modpath}-${name}
      DESTINATION "${PYTHON_INSTALL_PATH}/${modpath}"
      COMPONENT   runtime)
  endif (NOT ${no_install_plugin})

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
  set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${PYTHON_OUTPUT_PATH}/${modpath}/__init__.py")

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    if (WIN32)
      set(output_path "${PYTHON_OUTPUT_PATH}/${config}")
    endif (WIN32)

    file(WRITE "${output_path}/${modpath}/__init__.py"
"${copyright_header}
${import_lines}
")
    set_directory_properties(PROPERTIES
      ADDITIONAL_MAKE_CLEAN_FILES "${output_path}/${modpath}/__init__.py")
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

function (add_python_module path modpath module)
  string(REPLACE "/" "." safe_modpath ${modpath})

  vistk_configure_file(python-module-${module}
    "${CMAKE_CURRENT_SOURCE_DIR}/${path}"
    "${PYTHON_OUTPUT_PATH}/${modpath}/${module}.py"
    PYTHON_EXECUTABLE)

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    if (WIN32)
      set(output_path "${PYTHON_OUTPUT_PATH}/${config}")
    endif (WIN32)

    string(TOUPPER "${config}" upper_config)

    vistk_configure_file(python-module-${module}-${config}
      "${CMAKE_CURRENT_SOURCE_DIR}/${path}"
      "${output_path}/${modpath}/${module}.py"
      PYTHON_EXECUTABLE)
  endforeach (config)

  if (NOT ${no_install_plugin})
    install(
      FILES       "${PYTHON_OUTPUT_PATH}/${modpath}/${module}.py"
      DESTINATION "${PYTHON_INSTALL_PATH}/${modpath}"
      COMPONENT   runtime)
  endif (NOT ${no_install_plugin})
endfunction (add_python_module path modpath module)

add_subdirectory(image)
add_subdirectory(scoring)
add_subdirectory(modules)
add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
add_subdirectory(processes)
add_subdirectory(schedules)

if (VISTK_ENABLE_TESTING)
  add_subdirectory(test)
endif (VISTK_ENABLE_TESTING)

make_python_init_file(vistk)
#  image
#  pipeline
#  pipeline_util)
