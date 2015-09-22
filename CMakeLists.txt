###
#
# Top level CMake file for vital
#

cmake_minimum_required(VERSION 3.0)

project(VITAL)

include(CMakeDependentOption)

###
# VITAL version
#
set(VITAL_VERSION_MAJOR 0)
set(VITAL_VERSION_MINOR 1)
set(VITAL_VERSION_PATCH 0)
set(VITAL_VERSION "${VITAL_VERSION_MAJOR}.${VITAL_VERSION_MINOR}.${VITAL_VERSION_PATCH}")

##
# Look for and use Fletch to find dependencies
#
set( fletch_DIR "" CACHE PATH "Path to FLETCH packages" )
if ( IS_DIRECTORY ${fletch_DIR} )
  find_package( fletch NO_MODULE )
endif()

###
# project specific directories
#
set(VITAL_CMAKE_DIR            "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_MODULE_PATH      ${VITAL_CMAKE_DIR} ${CMAKE_MODULE_PATH})

# ================================================================
###
# Set options
option( VITAL_ENABLE_TESTS     "Enable testing" ON)
option( VITAL_ENABLE_TOOLS     "Enable tools" ON)

if (VITAL_ENABLE_TESTS)
  enable_testing()
  include(CTest)
  set(BUILD_TESTING    ON)
  mark_as_advanced(BUILD_TESTING)

endif()

option( VITAL_ENABLE_PYTHON   "Enable Python bindings" OFF )

###
# KWIVER CMake config
#
include(kwiver-utils)
include(vital-flags) # vital specific CXX flags

###
# External Dependencies and Modules
#
include( vital-depends )

# set the export group name for our package exports
set(kwiver_export_name
  vital_exports)

# ================================================================
# project global includes
include_directories("${VITAL_SOURCE_DIR}")
include_directories("${VITAL_BINARY_DIR}")

# ==================================================================

# Why would we ever want to build static?
option(VITAL_BUILD_SHARED "Build VITAL packages shared or not" ON)

set( LIB_SUFFIX "" CACHE STRING
  "Library directory suffix. e.g. suffix=\"vital\" will install libraries in \"libvital\" rather than \"lib\"")
mark_as_advanced( LIB_SUFFIX )

if (VITAL_ENABLE_PYTHON)
    find_package(PythonInterp)
    find_package(PythonLibs)
endif()

set(BUILD_SHARED_LIBS ${VITAL_BUILD_SHARED})

# installing system setup script
if (fletch_ROOT)
    set(SET_FLETCH_LD_LIBRARY_PATH "export LD_LIBRARY_PATH=${fletch_ROOT}/lib:$LD_LIBRARY_PATH")
    set(SET_FLETCH_PYTHON_PATH "export PYTHONPATH=${fletch_ROOT}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")
endif()

configure_file(${VITAL_CMAKE_DIR}/setup_VITAL.sh.in
  ${VITAL_BINARY_DIR}/setup_VITAL.sh )

# install set up script
install( PROGRAMS   ${VITAL_BINARY_DIR}/setup_VITAL.sh
  DESTINATION ${CMAKE_INSTALL_PREFIX} )

add_subdirectory( vital )

##################################################################
# Top level installation
#
set(kwiver_cmake_install_dir "lib${LIB_SUFFIX}/cmake/vital")

# Install rules for CMake utilities
include( kwiver-install-utils )

# Prepare space-separated list of library names for config
get_property(vital_libs GLOBAL PROPERTY kwiver_libraries)
string(REPLACE ";" " " vital_libs "${vital_libs}")

# Configure build-tree CMake config file and export associated targets file
set(VITAL_CONFIG_FILE         "${VITAL_BINARY_DIR}/vital-config.cmake")
set(VITAL_SOURCE_DIR           "${CMAKE_SOURCE_DIR}")
set(VITAL_BINARY_DIR           "${CMAKE_BINARY_DIR}")
set(module_path                "${VITAL_CMAKE_DIR}")

kwiver_configure_file(vital-config
  "${VITAL_SOURCE_DIR}/CMake/vital-config.cmake.in"
  "${VITAL_CONFIG_FILE}"
  VITAL_SOURCE_DIR
  VITAL_BINARY_DIR
  module_path
  vital_libs
  )
kwiver_export_targets("${VITAL_BINARY_DIR}/vital-config-targets.cmake")

# Configure install-tree CMake config file and export associated targets file
set(VITAL_CONFIG_INSTALL_FILE "${VITAL_BINARY_DIR}/vital-config-install.cmake")
kwiver_configure_file(vital-install-config
  "${VITAL_SOURCE_DIR}/CMake/vital-config-install.cmake.in"
  "${VITAL_CONFIG_INSTALL_FILE}"
  module_path
  vital_libs
  )

kwiver_install(
  FILES       "${VITAL_CONFIG_INSTALL_FILE}"
  DESTINATION "${kwiver_cmake_install_dir}"
  RENAME      vital-config.cmake
  )

kwiver_install(
  EXPORT      ${kwiver_export_name}
  DESTINATION "${kwiver_cmake_install_dir}"
  FILE        vital-config-targets.cmake
  )
