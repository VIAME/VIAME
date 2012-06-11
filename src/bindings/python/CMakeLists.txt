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
  # Not fixed yet.
  #if (${Boost_VERSION} LESS 10XX00)
    # Override Boost's override.hpp
    include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/helpers/include/exceptions_in_override")
  #endif (${Boost_VERSION} LESS 10XX00)
endif (VISTK_ENABLE_BOOST_PYTHON_FIXES)

add_subdirectory(helpers)

add_custom_target(python)

function (add_vistk_python_library name modpath)
  string(REPLACE "/" "." safe_modpath ${modpath})

  set(library_subdir /python${PYTHON_VERSION}${PYTHON_ABIFLAGS}/${modpath})
  set(component runtime)

  vistk_add_library(python-${safe_modpath}-${name}
    ${ARGN})

  set_target_properties(python-${safe_modpath}-${name}
    PROPERTIES
      OUTPUT_NAME ${name}
      PREFIX      "")

  if (WIN32)
    set_target_properties(python-${safe_modpath}-${name}
      PROPERTIES
        SUFFIX ".pyd")
  endif (WIN32)

  add_dependencies(python
    python-${safe_modpath}-${name})
endfunction (add_vistk_python_library name modpath)

function (_add_python_module path modpath module)
  string(REPLACE "/" "." safe_modpath ${modpath})

  set(python_sitepath)
  set(python_arch arch)

  if (WIN32)
    set(python_install_path bin)
  else (WIN32)
    if (python_noarch)
      set(python_install_path lib)
      set(python_sitepath /site-packages)
      set(python_arch noarch)
    else (python_noarch)
      set(python_install_path lib${LIB_SUFFIX})
    endif (python_noarch)
  endif (WIN32)

  vistk_configure_file(python-module-${python_arch}-${safe_modpath}-${module}
    "${path}"
    "${python_output_path}${sitepath}/${modpath}/${module}.py"
    PYTHON_EXECUTABLE)

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    vistk_configure_file(python-module-${python_arch}-${safe_modpath}-${module}-${config}
      "${path}"
      "${python_output_path}/${config}${sitepath}/${modpath}/${module}.py"
      PYTHON_EXECUTABLE)

    add_dependencies(python
      python-module-${python_arch}-${safe_modpath}-${module}-${config})
  endforeach (config)

  vistk_install(
    FILES       "${python_output_path}${sitepath}/${modpath}/${module}.py"
    DESTINATION "${python_install_path}/python${PYTHON_VERSION}${python_sitepath}/${modpath}"
    COMPONENT   runtime)

  add_dependencies(python
    python-module-${python_arch}-${safe_modpath}-${module})

  if (python_both_arch)
    set(python_both_arch)
    set(python_noarch TRUE)
    set(sitepath /site-packages)

    if (NOT WIN32)
      _add_python_module(
        "${path}"
        ${modpath}
        ${module})
    endif (NOT WIN32)
  endif (python_both_arch)
endfunction (_add_python_module path modpath module)

function (add_python_module path modpath module)
  _add_python_module("${CMAKE_CURRENT_SOURCE_DIR}/${path}"
    ${modpath}
    ${module})
endfunction (add_python_module path modpath module)

set(copyright_header
"#ckwg +4
# Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.")

function (make_python_init_file modpath)
  string(REPLACE "/" "." safe_modpath ${modpath})

  set(init_template "${CMAKE_CURRENT_BINARY_DIR}/${safe_modpath}.__init__.py")

  file(WRITE "${init_template}"
    "${copyright_header}\n\n")

  foreach (module ${ARGN})
    file(APPEND "${init_template}"
      "from ${module} import *\n")
  endforeach (module)

  _add_python_module("${init_template}"
    ${modpath}
    __init__)
endfunction (make_python_init_file modpath)

function (make_python_plugin_init_file modpath)
  string(REPLACE "/" "." safe_modpath ${modpath})

  set(init_template "${CMAKE_CURRENT_BINARY_DIR}/${safe_modpath}.__init__.py")

  file(WRITE "${init_template}"
    "${copyright_header}\n\n")
  file(APPEND "${init_template}"
    "from pkgutil import extend_path\n")
  file(APPEND "${init_template}"
    "__path__ = extend_path(__path__, __name__)\n")

  _add_python_module("${init_template}"
    ${modpath}
    __init__)
endfunction (make_python_plugin_init_file modpath)

add_subdirectory(vistk)
add_subdirectory(modules)
add_subdirectory(processes)
add_subdirectory(schedulers)

if (VISTK_ENABLE_TESTING)
  add_subdirectory(test)
endif (VISTK_ENABLE_TESTING)

set (python_both_arch TRUE)

make_python_init_file(vistk)
