###
#
# Top level CMake file for vital
#

cmake_minimum_required(VERSION 3.0)

project(KWIVER-VITAL)

include(CMakeDependentOption)

###
# VITAL version
#
# These symbols need to be "KWIVER" to work with the CMake infrastructure.
# Other symbole below too.
#
set(KWIVER_VERSION_MAJOR 0)
set(KWIVER_VERSION_MINOR 1)
set(KWIVER_VERSION_PATCH 0)
set(KWIVER_VERSION "${KWIVER_VERSION_MAJOR}.${KWIVER_VERSION_MINOR}.${KWIVER_VERSION_PATCH}")

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
set(KWIVER_PACKAGING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/VITAL_packaging")
set(KWIVER_PACKAGING_BINARY_DIR "${CMAKE_BINARY_DIR}/VITAL_packaging")
set(KWIVER_CMAKE_DIR            "${CMAKE_SOURCE_DIR}/CMake")

# root directories
set(KWIVER_SOURCE_DIR           "${CMAKE_SOURCE_DIR}")
set(KWIVER_BINARY_DIR           "${CMAKE_BINARY_DIR}")

set(CMAKE_MODULE_PATH      ${KWIVER_CMAKE_DIR} ${CMAKE_MODULE_PATH})

# ================================================================
###
# Set options
set(KWIVER_LIB_SUFFIX "" CACHE STRING "String suffix appended to the library directory name we install into.")

option(KWIVER_ENABLE_TESTS     "Enable testing" ON)

if (KWIVER_ENABLE_TESTS)
  enable_testing()
  include(CTest)
  set(BUILD_TESTING    ON)
  mark_as_advanced(BUILD_TESTING)

endif()

if (KWIVER_ENABLE_PYTHON)
  set(vital_boost_python_package python)
endif()

###
# KWIVER CMake config
#
include(kwiver-utils)
include(kwiver-flags)

###
# External Dependencies and Modules
#
include( vital-depends )

# set the export group name for our package exports
set(kwiver_export_name
  vital_exports)

# define which boost we are using
if (DEFINED BOOST_ROOT)
  set(BOOST_ROOT_ARG ${BOOST_ROOT})
else()
  set(BOOST_ROOT_ARG ${Boost_LIBRARY_DIR}/..)
endif()

# ================================================================
# project global includes
include_directories("${KWIVER_SOURCE_DIR}")
include_directories("${KWIVER_BINARY_DIR}")

# ==================================================================

# Why would we ever want to build static?
option(KWIVER_BUILD_SHARED "Build VITAL packages shared or not" ON)

if (KWIVER_ENABLE_PYTHON)
    find_package(PythonInterp)
    find_package(PythonLibs)
endif()

if(KWIVER_BUILD_SHARED)
  set(LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
  set(LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
else()
  set(LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
  set(LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

set(BUILD_SHARED_LIBS ${KWIVER_BUILD_SHARED})

# installing system setup script
if (fletch_ROOT)
    set(SET_FLETCH_LD_LIBRARY_PATH "export LD_LIBRARY_PATH=${fletch_ROOT}/lib:$LD_LIBRARY_PATH")
    set(SET_FLETCH_PYTHON_PATH "export PYTHONPATH=${fletch_ROOT}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages")
endif()

configure_file(${KWIVER_CMAKE_DIR}/setup_VITAL.sh.in
  ${CMAKE_INSTALL_PREFIX}/setup_VITAL.sh )

add_subdirectory( vital )

##################################################################
# Top level installation
#
set(kwiver_cmake_install_dir "lib${KWIVER_LIB_SUFFIX}/CMake/vital")

# Install rules for CMake utilities
## include( kwiver-install-utils )

# Prepare space-separated list of library names for config
get_property(vital_libs GLOBAL PROPERTY kwiver_libraries)
string(REPLACE ";" " " vital_libs "${vital_libs}")

# Configure build-tree CMake config file and export associated targets file
set(KWIVER_CONFIG_FILE "${KWIVER_BINARY_DIR}/vital-config.cmake")

kwiver_configure_file(vital-config
  "${KWIVER_SOURCE_DIR}/CMake/vital-config.cmake.in"
  "${KWIVER_CONFIG_FILE}"
  KWIVER_SOURCE_DIR
  KWIVER_BINARY_DIR
  EIGEN3_INCLUDE_DIR
  vital_libs
  )
kwiver_export_targets("${KWIVER_BINARY_DIR}/vital-config-targets.cmake")

# Configure install-tree CMake config file and export associated targets file
set(KWIVER_CONFIG_INSTALL_FILE "${KWIVER_BINARY_DIR}/vital-config-install.cmake")
kwiver_configure_file(vital-install-config
  "${KWIVER_SOURCE_DIR}/CMake/vital-config-install.cmake.in"
  "${KWIVER_CONFIG_INSTALL_FILE}"
  EIGEN3_INCLUDE_DIR
  vital_libs
  )

kwiver_install(
  FILES       "${KWIVER_CONFIG_INSTALL_FILE}"
  DESTINATION "${kwiver_cmake_install_dir}"
  RENAME      vital-config.cmake
  )

kwiver_install(
  EXPORT      ${kwiver_export_name}
  DESTINATION "${kwiver_cmake_install_dir}"
  FILE        vital-config-targets.cmake
  )
