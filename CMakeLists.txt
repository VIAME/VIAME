###
# Top level build script for kwiver-build
#
cmake_minimum_required(VERSION 3.0)

project(KWIVER)

include(CMakeDependentOption)

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
set(KWIVER_PACKAGES_DIR         "${CMAKE_SOURCE_DIR}/Packages") ## +delete this symbol
set(KWIVER_BUILD_PREFIX         "${CMAKE_BINARY_DIR}/build") ## +delete this symbol
set(KWIVER_BUILD_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install") ## +delete this symbol

set(KWIVER_PACKAGING_SOURCE_DIR "${CMAKE_SOURCE_DIR}/KWIVER_packaging")
set(KWIVER_PACKAGING_BINARY_DIR "${CMAKE_BINARY_DIR}/KWIVER_packaging")
set(KWIVER_CMAKE_DIR            "${CMAKE_SOURCE_DIR}/CMake")

# root directories
set(KWIVER_SOURCE_DIR           "${CMAKE_SOURCE_DIR}")
set(KWIVER_BINARY_DIR           "${CMAKE_BINARY_DIR}")

set(KWIVER_CONFIG_FILE          "${KWIVER_BINARY_DIR}/kwiver-config.cmake")

set(CMAKE_MODULE_PATH           ${KWIVER_CMAKE_DIR} ${CMAKE_MODULE_PATH})

set(CMAKE_FIND_ROOT_PATH   ${KWIVER_BUILD_INSTALL_PREFIX} ${CMAKE_FIND_ROOT_PATH})  ## +delete this symbol
set(CMAKE_LIBRARY_PATH     ${KWIVER_BUILD_INSTALL_PREFIX}/lib)  ## +delete this symbol
set(KWIVER_LINKER_FLAGS    "-L${CMAKE_LIBRARY_PATH}" ) ## +delete this symbol

# ================================================================
option(KWIVER_ENABLE_OPENCV    "Enable OpenCV-dependent code" ON )

option(KWIVER_ENABLE_TESTS     "Enable testing" ON)

if (KWIVER_ENABLE_TESTS)
  enable_testing()
  include(CTest)
  set(BUILD_TESTING    ON)
  mark_as_advanced(BUILD_TESTING)
endif()

option( KWIVER_ENABLE_PYTHON    "Enable python code" OFF )

if (KWIVER_ENABLE_PYTHON)
  set(kwiver_boost_python_package python)
endif()

###
# KWIVER CMake config
#
include(kwiver-utils)

#
# Be careful establishing compiler/linker options at this
# level. Component packages may not perform well (or build at all)
# with aggressive warnings.
#

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
include_directories(BEFORE "${KWIVER_BUILD_INSTALL_PREFIX}/include") ## +delete this symbol
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

configure_file(${KWIVER_CMAKE_DIR}/setup_KWIVER.sh.in
  ${KWIVER_BUILD_INSTALL_PREFIX}/setup_KWIVER.sh )

#include_directories("${KWSYS_INCLUDE_DIR}")


# ================================================================
# Setting up external projects for KWIVER components
#
include(ExternalProject)
set_property(DIRECTORY PROPERTY EP_STEP_TARGETS configure build install)

# Common CMake args to send to all external projects
set(KWIVER_ARGS_COMMON_CMAKE
  # CMAKE options
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX:PATH=${KWIVER_BUILD_INSTALL_PREFIX}
  -DBUILD_SHARED_LIBS:BOOL=${KWIVER_BUILD_SHARED}
  -DCMAKE_FIND_ROOT_PATH:PATH=${CMAKE_FIND_ROOT_PATH}
  -DCMAKE_LIBRARY_PATH:PATH=${CMAKE_LIBRARY_PATH}
  -DKWIVER_ENABLE_OPENCV:BOOL=${KWIVER_ENABLE_OPENCV}
  -DKWIVER_ENABLE_DOCS:BOOL=${KWIVER_ENABLE_DOCS}
  -DKWIVER_ENABLE_TESTS:BOOL=${KWIVER_ENABLE_TESTS}
  -DEIGEN3_ROOT:PATH=${EIGEN3_ROOT}
  -Dkwiver_DIR:PATH=${KWIVER_BINARY_DIR}
  -Dfletch_DIR:PATH=${fletch_DIR}
  -vital_DIR:PATH=${vital_DIR}
  -DPROJ4_ROOT:PATH=${PROJ4_ROOT}
  -DKML_DIR:PATH=${LIBKML_ROOT}/lib/cmake
  -DGeographicLib_ROOT:PATH=${GeographicLib_ROOT}
  -DTinyXML_ROOT:PATH=${TinyXML_ROOT}
  -DLIBJSON_ROOT:PATH=${LIBJSON_ROOT}
  -DSHAPELIB_ROOT:PATH=${SHAPELIB_ROOT}
)

set(KWIVER_ARGS_COMMON_COMPILER
  # Compiler options
  -DCMAKE_CXX_COMPILER_ARG1:STRING=${CMAKE_CXX_COMPILER_ARG1}
  -DCMAKE_C_COMPILER_ARG1:STRING=${CMAKE_C_COMPILER_ARG1}
  -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
  -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
  -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
  -DCMAKE_CXX_FLAGS_MINSIZEREL:STRING=${CMAKE_CXX_FLAGS_MINSIZEREL}
  -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
  -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
  -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
  -DCMAKE_C_FLAGS_MINSIZEREL:STRING=${CMAKE_C_FLAGS_MINSIZEREL}
  -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
  -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_C_FLAGS_RELWITHDEBINFO}
  )

set(KWIVER_ARGS_COMMON_BOOST
  # Boost files and options
  -DBOOST_ROOT=${BOOST_ROOT_ARG}
  -DBoost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}
  -DBoost_LIBRARY_DIR=${Boost_LIBRARY_DIRS}
)

set( KWIVER_ARGS_OPENCV )

if( KWIVER_ENABLE_OPENCV )
  set( KWIVER_ARGS_OPENCV
    -DOpenCV_DIR:PATH=${OpenCV_CONFIG_PATH}
    )
endif()

set(KWIVER_ARGS_COMMON_LINKER
  # Linker options
  "-DCMAKE_EXE_LINKER_FLAGS:STRING=${KWIVER_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}"
  "-DCMAKE_MODULE_LINKER_FLAGS:STRING=${KWIVER_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}"
  "-DCMAKE_SHARED_LINKER_FLAGS:STRING=${KWIVER_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}"
  )

set(KWIVER_ARGS_COMMON ${KWIVER_ARGS_COMMON_CMAKE}
                       ${KWIVER_ARGS_OPENCV}
                       ${KWIVER_ARGS_COMMON_COMPILER}
                       ${KWIVER_ARGS_COMMON_LINKER}
                       ${KWIVER_ARGS_COMMON_BOOST}
                       ${KWIVER_ARGS_COMMON_EIGEN}
  )

#
# Work out which Packages to build based on user choice and their
# interdependencies
#
option(KWIVER_ENABLE_ALL_PACKAGES "Should KWIVER build all of the optional packages?" OFF)
# Leave as advanced until VIVIA and VIBRANT aren't advanced
mark_as_advanced(KWIVER_ENABLE_ALL_PACKAGES)
option(KWIVER_ENABLE_PROCESSES "Should the KWIVER Sprokit Processes Be Built?" OFF)
option(KWIVER_ENABLE_MAPTK "Enable building MapTK" FALSE)
option(KWIVER_ENABLE_ARROWS "Enable building arrows" FALSE)
option(KWIVER_ENABLE_VIBRANT "Enable building VIBRANT" FALSE)
mark_as_advanced(KWIVER_ENABLE_VIBRANT)
option(KWIVER_ENABLE_KWANT "Enable building KWANT" FALSE)
mark_as_advanced(KWIVER_ENABLE_KWANT)
option(KWIVER_ENABLE_VIVIA "Enable building ViVIA" FALSE)
mark_as_advanced(KWIVER_ENABLE_VIVIA)


foreach(package MAPTK VIBRANT KWANT VIVIA PROCESSES)
  if (KWIVER_ENABLE_ALL_PACKAGES)
      set(KWIVER_ENABLE_${package} TRUE CACHE BOOL "" FORCE)
  endif()
endforeach()
#
set(KWIVER_ENABLE_ALL_PACKAGES FALSE CACHE BOOL "" FORCE)

if (KWIVER_ENABLE_VIVIA)
    set(KWIVER_ENABLE_KWANT TRUE CACHE BOOL "" FORCE)
    set(KWIVER_ENABLE_VIBRANT TRUE CACHE BOOL "" FORCE)
endif()

#
# Which applications to enable in ViVIA
#
set(KWIVER_ARGS_VIVIA_APPS
    "-DVISGUI_ENABLE_VIQUI:BOOL=FALSE"
    "-DVISGUI_ENABLE_VPVIEW:BOOL=FALSE"
    "-DVISGUI_ENABLE_VSPLAY:BOOL=TRUE"
    )

set(KWIVER_ARGS_VIVIA_DEPS
    "-DQT_QMAKE_EXECUTABLE:PATH=${QT_QMAKE_EXECUTABLE}"
    "-DVTK_DIR:PATH=${VTK_ROOT}/lib/cmake/vtk-6.2"
    )

# ================================================================
# Adding external components
#
# include(External_VXL)

if (KWIVER_ENABLE_MAPTK)
    include(External_maptk)
endif()

if (KWIVER_ENABLE_VIBRANT)
    include(External_vibrant)
endif()

if (KWIVER_ENABLE_KWANT)
	include(External_kwant)
endif()

if (KWIVER_ENABLE_VIVIA)
    include(External_vivia)
endif()

include_directories( SYSTEM ${CMAKE_CURRENT_SOURCE_DIR} )


# ================================================================
link_directories( ${KWIVER_BUILD_INSTALL_PREFIX}/lib )

# ==================================================================
###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()

##
# System specific compiler flags
include(kwiver-flags)

##
# check compiler support
include( kwiver-configcheck )

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
# vital includes are relative to ${CMAKE_CURRENT_SOURCE_DIR}

# TODO enforce some rational requirements.

# sprokit requires vital
# arrows.processes requires sprokit

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
