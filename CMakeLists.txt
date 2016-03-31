#
# Top level Cmake file for arrows
#
cmake_minimum_required(VERSION 3.0)

project(ARROWS)

include(CMakeDependentOption)

###
# ARROWS version
set(ARROWS_VERSION_MAJOR 0)
set(ARROWS_VERSION_MINOR 1)
set(ARROWS_VERSION_PATCH 0)
set(ARROWS_VERSION "${ARROWS_VERSION_MAJOR}.${ARROWS_VERSION_MINOR}.${ARROWS_VERSION_PATCH}")

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
set(ARROWS_SOURCE_DIR           "${CMAKE_SOURCE_DIR}")
set(ARROWS_BINARY_DIR           "${CMAKE_BINARY_DIR}")
set(ARROWS_CMAKE_DIR            "${CMAKE_SOURCE_DIR}/CMake")

set(CMAKE_MODULE_PATH      ${ARROWS_CMAKE_DIR} ${ARROWS_CMAKE_DIR}/modules ${CMAKE_MODULE_PATH})

# ================================================================
###
# Set options
option( ARROWS_ENABLE_TESTS     "Enable testing" ON)
option( ARROWS_ENABLE_TOOLS     "Enable tools" ON)
option( ARROWS_ENABLE_PYTHON    "Enable python code" OFF )

###
# External Dependencies and Modules
#
include( arrows-depends )


if (ARROWS_ENABLE_TESTS)
  enable_testing()
  include(CTest)
  set(BUILD_TESTING    ON)
  mark_as_advanced(BUILD_TESTING)
endif()

###
# ARROWS CMake config
#
include(kwiver-utils)
include(arrows-utils)
include(vital-flags) # arrows specific CXX flags

##
# check compiler support
include( kwiver-configcheck )

# set the export group name for our package exports
set(kwiver_export_name   arrows_exports)

# ================================================================
# project global includes
include_directories("${ARROWS_SOURCE_DIR}")
include_directories("${ARROWS_BINARY_DIR}")

set( LIB_SUFFIX "" CACHE STRING
  "Library directory suffix. e.g. suffix=\"arrows\" will install libraries in \"libarrows\" rather than \"lib\"")
mark_as_advanced( LIB_SUFFIX )

set(BUILD_SHARED_LIBS TRUE)

if (ARROWS_ENABLE_PYTHON)
  # installing system setup script
  find_package(PythonInterp)      # needed for PYTHON_VERSION_MAJOR et al
  if (fletch_ROOT)
    set(SET_FLETCH_LD_LIBRARY_PATH "export LD_LIBRARY_PATH=${fletch_ROOT}/lib:$LD_LIBRARY_PATH")
    set(SET_FLETCH_PYTHON_PATH "export PYTHONPATH=${fletch_ROOT}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages:$PYTHONPATH")
  endif()
endif()

##
# Create initial setup script
set(ARROWS_SETUP_SCRIPT_FILE  ${ARROWS_BINARY_DIR}/setup_arrows.sh )

configure_file(
  ${ARROWS_CMAKE_DIR}/setup_arrows.sh.in
  ${ARROWS_SETUP_SCRIPT_FILE}
  )

# install set up script
install( PROGRAMS   ${ARROWS_SETUP_SCRIPT_FILE}
  DESTINATION ${CMAKE_INSTALL_PREFIX} )

add_subdirectory( arrows )

##################################################################
# Top level installation
#
set(kwiver_cmake_install_dir "lib${LIB_SUFFIX}/cmake/arrows")

# Install rules for CMake utilities
include( kwiver-install-utils )

# Prepare space-separated list of library names for config
get_property(arrows_libs GLOBAL PROPERTY kwiver_libraries)
string(REPLACE ";" " " arrows_libs "${arrows_libs}")

# Configure build-tree CMake config file and export associated targets file
set(ARROWS_CONFIG_FILE         "${ARROWS_BINARY_DIR}/arrows-config.cmake")
set(module_path                "${ARROWS_CMAKE_DIR}")

kwiver_configure_file(arrows-config
  "${ARROWS_SOURCE_DIR}/CMake/arrows-config.cmake.in"
  "${ARROWS_CONFIG_FILE}"
  ARROWS_SOURCE_DIR
  ARROWS_BINARY_DIR
  module_path
  arrows_libs
  )
kwiver_export_targets("${ARROWS_BINARY_DIR}/arrows-config-targets.cmake")

# Configure install-tree CMake config file and export associated targets file
set(ARROWS_CONFIG_INSTALL_FILE "${ARROWS_BINARY_DIR}/arrows-config-install.cmake")
kwiver_configure_file(arrows-install-config
  "${ARROWS_SOURCE_DIR}/CMake/arrows-config-install.cmake.in"
  "${ARROWS_CONFIG_INSTALL_FILE}"
  module_path
  arrows_libs
  )

kwiver_install(
  FILES       "${ARROWS_CONFIG_INSTALL_FILE}"
  DESTINATION "${kwiver_cmake_install_dir}"
  RENAME      arrows-config.cmake
  )

kwiver_install(
  EXPORT      ${kwiver_export_name}
  DESTINATION "${kwiver_cmake_install_dir}"
  FILE        arrows-config-targets.cmake
  )
