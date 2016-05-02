###
# Top level build script for kwiver-build
#
cmake_minimum_required(VERSION 3.0)

project(KWIVER)

###
# KWIVER version
set(KWIVER_VERSION_MAJOR 0)
set(KWIVER_VERSION_MINOR 1)
set(KWIVER_VERSION_PATCH 0)
set(KWIVER_VERSION "${KWIVER_VERSION_MAJOR}.${KWIVER_VERSION_MINOR}.${KWIVER_VERSION_PATCH}")

###
# Look for and use Fletch to find dependencies
#
set( fletch_DIR "" CACHE PATH "Path to FLETCH packages" )
if ( IS_DIRECTORY ${fletch_DIR} )
  find_package( fletch NO_MODULE )
endif()

if (NOT IS_ABSOLUTE "${CMAKE_INSTALL_PREFIX}")
  message(FATAL_ERROR "KWIVER does not support a relative install prefix (${CMAKE_INSTALL_PREFIX}).")
endif ()

###
# project specific directories
#
set(KWIVER_CMAKE_DIR            "${CMAKE_SOURCE_DIR}/CMake")

# root directories
set(KWIVER_SOURCE_DIR           "${CMAKE_SOURCE_DIR}")
set(KWIVER_BINARY_DIR           "${CMAKE_BINARY_DIR}")

set(KWIVER_CONFIG_FILE          "${KWIVER_BINARY_DIR}/kwiver-config.cmake")

set(CMAKE_MODULE_PATH           ${KWIVER_CMAKE_DIR} ${CMAKE_MODULE_PATH})

# ================================================================
option(KWIVER_ENABLE_OPENCV    "Enable OpenCV-dependent code" ON )

option(KWIVER_ENABLE_TESTS     "Enable testing" ON)

if (KWIVER_ENABLE_TESTS)
  enable_testing()
  include(CTest)
  set(BUILD_TESTING    ON)
  mark_as_advanced(BUILD_TESTING)
endif()

## TODO: How should pytgon and c lib interact?
option( KWIVER_C_LIB     "Enable C bindings libraries" OFF)
option( KWIVER_ENABLE_PYTHON    "Enable python code" OFF )

if (KWIVER_ENABLE_PYTHON)
  if ( NOT KWIVER_C_LIB )
    message( SEND_ERROR  "Python can not be enabled unless KWIVER_C_LIB is also enabled." )
  endif()
  set(kwiver_boost_python_package     python)
endif()

###
# KWIVER CMake config
#
include(kwiver-utils)

##
# System specific compiler flags
include(kwiver-flags)

##
# check compiler support
include( kwiver-configcheck )


# set the name for our package exports
set(kwiver_export_name      kwiver_exports)

###
# External Dependencies and Modules
# Will be supplied by Fletch
include( kwiver-depends )

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
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# kwiversys includes are generated in the binary directory
include_directories("${KWIVER_BINARY_DIR}/vital/kwiversys")

# ==================================================================
# Why would we ever want to build static?
option(KWIVER_BUILD_SHARED "Build KWIVER packages shared or not" ON)

if (KWIVER_ENABLE_PYTHON)
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs)

    # to be compatible with sprokit
    set(PYTHON_VERSION "2.7"
    CACHE STRING "The version of python to use for bindings")
  set(PYTHON_ABIFLAGS ""
    CACHE STRING "The ABI flags for the version of Python being used")

  include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

endif(KWIVER_ENABLE_PYTHON)

set(BUILD_SHARED_LIBS ${KWIVER_BUILD_SHARED})

#include_directories("${KWSYS_INCLUDE_DIR}")

option(KWIVER_ENABLE_PROCESSES "Should the KWIVER Sprokit Processes Be Built?" OFF)
option(KWIVER_ENABLE_ARROWS "Enable building arrows" FALSE)

include_directories( SYSTEM ${CMAKE_CURRENT_SOURCE_DIR} )


# ================================================================
link_directories( ${KWIVER_BUILD_INSTALL_PREFIX}/lib )

# ==================================================================
###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()

# ==================================================================

set( LIB_SUFFIX "" CACHE STRING
  "Library directory suffix. e.g. suffix=\"kwiver\" will install libraries in \"libkwiver\" rather than \"lib\"")
mark_as_advanced( LIB_SUFFIX )


# Create initial setup script
set(KWIVER_SETUP_SCRIPT_FILE  ${KWIVER_BINARY_DIR}/setup_KWIVER.sh )

configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.sh.in
  ${KWIVER_SETUP_SCRIPT_FILE}
  )

# install set up script
install( PROGRAMS   ${KWIVER_SETUP_SCRIPT_FILE}
  DESTINATION ${CMAKE_INSTALL_PREFIX} )

add_subdirectory( vital )

## if sprokit enabled
option(KWIVER_ENABLE_SPROKIT "Enable building sprokit" ON )

## If arrows enabled
option(KWIVER_ENABLE_ARROWS "Enable building arrows" ON )

if (KWIVER_ENABLE_SPROKIT)
  add_subdirectory(sprokit)
  include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/sprokit/src" )
  include_directories( "${CMAKE_CURRENT_BINARY_DIR}/sprokit/src" ) # for generated files
endif()

if (KWIVER_ENABLE_ARROWS)
  add_subdirectory( arrows )
endif()

##################################################################
# Top level installation
#
set(kwiver_cmake_install_dir lib${LIB_SUFFIX}/cmake/kwiver)

# Install rules for CMake utilities
include( kwiver-install-utils )

# Prepare space-separated list of library names for config
get_property(kwiver_libs GLOBAL PROPERTY kwiver_libraries)
string(REPLACE ";" " " kwiver_libs "${kwiver_libs}")

# Configure build-tree CMake config file and export associated targets file
set(KWIVER_LIBRARY_DIR "${KWIVER_BINARY_DIR}/lib") ##? TEMP - not used

kwiver_configure_file(kwiver-config
  "${KWIVER_SOURCE_DIR}/CMake/kwiver-config.cmake.in"
  "${KWIVER_CONFIG_FILE}"
  KWIVER_SOURCE_DIR
  KWIVER_BINARY_DIR
  EIGEN3_INCLUDE_DIR
  kwiver_libs
  )
kwiver_export_targets("${KWIVER_BINARY_DIR}/kwiver-config-targets.cmake")

# Configure install-tree CMake config file and export associated targets file
set(KWIVER_CONFIG_INSTALL_FILE "${KWIVER_BINARY_DIR}/kwiver-config-install.cmake")
kwiver_configure_file(kwiver-install-config
  "${KWIVER_SOURCE_DIR}/CMake/kwiver-config-install.cmake.in"
  "${KWIVER_CONFIG_INSTALL_FILE}"
  EIGEN3_INCLUDE_DIR
  kwiver_libs
  )

kwiver_install(
  FILES       "${KWIVER_CONFIG_INSTALL_FILE}"
  DESTINATION "${kwiver_cmake_install_dir}"
  RENAME      kwiver-config.cmake
  )

kwiver_install(
  EXPORT      ${kwiver_export_name}
  DESTINATION "${kwiver_cmake_install_dir}"
  FILE        kwiver-config-targets.cmake
  )


# ==================================================================
# CPack Packaging
#

#TODO: Define package dependencies
set(KWIVER_DEPS "")

if(EXISTS /etc/redhat-release)
  file(READ /etc/redhat-release RHEL_VERSION)
  string(REGEX REPLACE ".*release ([^\\. ]*).*" "\\1" RHEL_VERSION "${RHEL_VERSION}")
  set(CPACK_SYSTEM_NAME "el${RHEL_VERSION}.${CMAKE_SYSTEM_PROCESSOR}")
  set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")
  set(CPACK_RPM_PACKAGE_REQUIRES "${KWIVER_DEPS}")
else()
  set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CPACK_PACKAGE_NAME              "KWIVER")
set(CPACK_PACKAGE_VENDOR            "Kitware, Inc.")
set(CPACK_PACKAGE_CONTACT           "kitware@kitware.com")
set(CPACK_MONOLITHIC_INSTALL        true)
set(CPACK_PACKAGE_VERSION_MAJOR     "${KWIVER_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR     "${KWIVER_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH     "${KWIVER_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION           "${KWIVER_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE     "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGING_INSTALL_PREFIX  "${CMAKE_INSTALL_PREFIX}")
set(CPACK_PACKAGE_FILE_NAME         "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}")

include (CPack)
