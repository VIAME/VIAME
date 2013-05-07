project(sprokit_python)

set(boost_python_library
  python)

if (SPROKIT_ENABLE_PYTHON3)
  set(boost_python_library
    python3)
endif ()

string(TOUPPER ${boost_python_library} boost_python_library_upper)
set(boost_python_library_var
  Boost_${boost_python_library_upper}_LIBRARY)

find_package(Boost ${sprokit_boost_version} REQUIRED
  COMPONENTS
    iostreams
    ${boost_python_library})

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

set(boost_header_targets)

option(SPROKIT_ENABLE_BOOST_PYTHON_FIXES "Enables fixes for Boost.Python bugs (recommended)" ON)
mark_as_advanced(SPROKIT_ENABLE_BOOST_PYTHON_FIXES)
if (SPROKIT_ENABLE_BOOST_PYTHON_FIXES)
  # Not fixed yet.
  #if (Boost_VERSION LESS 10XX00)
    # Override Boost's invoke.hpp
    set(reldir "helpers/include/threading")
    set(target_name "boost-python-invoke.hpp")
    sprokit_configure_directory(${target_name}
      "${CMAKE_CURRENT_SOURCE_DIR}/${reldir}/boost"
      "${CMAKE_CURRENT_BINARY_DIR}/${reldir}/${BOOST_MANGLE_NAMESPACE}")
    if (NOT BOOST_MANGLE_NAMESPACE STREQUAL "boost")
      include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/${reldir}")
    endif ()
    include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/${reldir}")
    set(boost_header_targets
      ${boost_header_targets}
      ${target_name})
  #endif ()
  # Not fixed yet.
  #if (Boost_VERSION LESS 10XX00)
    # Override Boost's override.hpp
    set(reldir "helpers/include/exceptions_in_override")
    set(target_name "boost-python-override.hpp")
    sprokit_configure_directory(${target_name}
      "${CMAKE_CURRENT_SOURCE_DIR}/${reldir}/boost"
      "${CMAKE_CURRENT_BINARY_DIR}/${reldir}/${BOOST_MANGLE_NAMESPACE}")
    if (NOT BOOST_MANGLE_NAMESPACE STREQUAL "boost")
      include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/${reldir}")
    endif ()
    include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_BINARY_DIR}/${reldir}")
    set(boost_header_targets
      ${boost_header_targets}
      ${target_name})
  #endif ()
endif ()

add_subdirectory(helpers)

add_custom_target(python)

function (_sprokit_add_python_library name modpath)
  _sprokit_create_safe_modpath(${modpath} safe_modpath)

  sprokit_add_python_library(${name} ${modpath}
    ${ARGN})

  if (boost_header_targets)
    add_dependencies(python-${safe_modpath}-${name}
      ${boost_header_targets})
  endif (boost_header_targets)
endfunction ()

set(copyright_header
"#ckwg +4
# Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.")

source_group("Python Files"
  REGULAR_EXPRESSION ".*\\.py\\.in$")
source_group("Python Files"
  REGULAR_EXPRESSION ".*\\.py$")

add_subdirectory(sprokit)
add_subdirectory(modules)
add_subdirectory(processes)
add_subdirectory(schedulers)

if (SPROKIT_ENABLE_TESTING)
  add_subdirectory(test)
endif ()

set (python_both_arch TRUE)

sprokit_create_python_init(sprokit)
