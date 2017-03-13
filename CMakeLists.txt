###
# Top level build script for kwiver-build
#
cmake_minimum_required(VERSION 3.3)

project(KWIVER)

include(CMakeDependentOption)

###
# KWIVER version
set(KWIVER_VERSION_MAJOR 1)
set(KWIVER_VERSION_MINOR 0)
set(KWIVER_VERSION_PATCH 0)
set(KWIVER_VERSION "${KWIVER_VERSION_MAJOR}.${KWIVER_VERSION_MINOR}.${KWIVER_VERSION_PATCH}")

###
# Look for and use Fletch to find dependencies
#
set( fletch_DIR "" CACHE PATH "Path to FLETCH packages" )
if ( IS_DIRECTORY ${fletch_DIR} )
  find_package( fletch NO_MODULE )
  # Search for libraries in the fletch install root
  list(APPEND CMAKE_PREFIX_PATH "${fletch_ROOT}")
endif()

# Fletch provides the PROJ4 symbols. We need the PROJ symbols.
if (PROJ4_ROOT)
  set(PROJ_ROOT "${PROJ4_ROOT}")
endif()

###
# project specific directories
##
set(KWIVER_CMAKE_DIR            "${CMAKE_SOURCE_DIR}/CMake")

# root directories
set(KWIVER_SOURCE_DIR           "${CMAKE_SOURCE_DIR}")
set(KWIVER_BINARY_DIR           "${CMAKE_BINARY_DIR}")

#
# kwiver module config file.
set(KWIVER_CONFIG_FILE          "${KWIVER_BINARY_DIR}/kwiver-config.cmake")
# Create initial setup script
set(KWIVER_SETUP_SCRIPT_FILE    "${KWIVER_BINARY_DIR}/setup_KWIVER.sh" )

LIST( INSERT CMAKE_MODULE_PATH  0         ${KWIVER_CMAKE_DIR} ) # prepend out cmake resources

# ================================================================
# project global includes
include_directories("${KWIVER_SOURCE_DIR}")
include_directories("${KWIVER_SOURCE_DIR}/sprokit/src")
include_directories("${KWIVER_BINARY_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# include vital internal headers for private use only
include_directories("${KWIVER_SOURCE_DIR}/vital/internal")

# kwiversys includes are generated in the binary directory
include_directories("${KWIVER_BINARY_DIR}/vital/kwiversys")

# ================================================================

include_directories( SYSTEM ${CMAKE_CURRENT_SOURCE_DIR} )

# ================================================================
###
# User options
OPTION(KWIVER_BUILD_SHARED     "Build KWIVER components shared or not" TRUE )
MARK_AS_ADVANCED( KWIVER_BUILD_SHARED )

OPTION(KWIVER_ENABLE_TESTS     "Enable testing" OFF )
OPTION(KWIVER_ENABLE_ARROWS    "Enable building arrows" OFF )
OPTION(KWIVER_ENABLE_TOOLS     "Enable tools" OFF )
OPTION(KWIVER_ENABLE_TRACK_ORACLE  "Enable track_oracle" OFF )

# if sprokit enabled
OPTION(KWIVER_ENABLE_SPROKIT "Enable building sprokit" OFF )

CMAKE_DEPENDENT_OPTION(KWIVER_ENABLE_PROCESSES "Should the KWIVER Sprokit Processes Be Built?" OFF
  "KWIVER_ENABLE_ARROWS AND  KWIVER_ENABLE_SPROKIT" OFF )

OPTION(KWIVER_USE_BUILD_TREE "Whether to include the build-tree plugin directory to the default plugin path list" OFF )
MARK_AS_ADVANCED(KWIVER_USE_BUILD_TREE)

option( KWIVER_ENABLE_EXTRAS "Enable extras content" OFF )

## TODO: How should pytgon and c lib interact?
OPTION( KWIVER_ENABLE_C_BINDINGS     "Enable C bindings libraries" OFF)
CMAKE_DEPENDENT_OPTION( KWIVER_ENABLE_PYTHON    "Enable python code" OFF
  KWIVER_ENABLE_C_BINDINGS OFF )

if (KWIVER_ENABLE_PYTHON)
  if ( NOT KWIVER_ENABLE_C_BINDINGS )
    message( SEND_ERROR  "Python can not be enabled unless KWIVER_ENABLE_C_BINDINGS is also enabled." )
  endif()
  set(kwiver_boost_python_package     python)
endif()

set(BUILD_SHARED_LIBS ${KWIVER_BUILD_SHARED})

# ==================================================================
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
# External Dependencies and Modules - Most Will be supplied by Fletch
include( kwiver-depends )

# define which boost we are using
if (DEFINED BOOST_ROOT)
  set(BOOST_ROOT_ARG ${BOOST_ROOT})
else()
  set(BOOST_ROOT_ARG ${Boost_LIBRARY_DIR}/..)
endif()

# ================================================================
link_directories( ${KWIVER_BINARY_DIR}/lib )

# ================================================================
###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  enable_testing()
  include(CTest)
  set(BUILD_TESTING    ON)  # needed for compatibility with kwiversys
  mark_as_advanced(BUILD_TESTING)

  add_subdirectory( tests )
  include_directories( tests )
endif()

if (KWIVER_ENABLE_PYTHON)
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs)

    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "import distutils.sysconfig; print distutils.sysconfig.get_python_lib(prefix='')"
      RESULT_VARIABLE proc_success
      OUTPUT_VARIABLE python_site_packages
      )

    if(NOT ${proc_success} EQUAL 0)
      message(FATAL_ERROR "Request for python site-packages location failed with error code: ${proc_success}")
    else()
      string(STRIP "${python_site_packages}" python_site_packages)
      message(STATUS "Python site-packages to install into: ${python_site_packages}")
    endif()

    # TODO Get version from above
    set(PYTHON_VERSION "2.7"
      CACHE STRING "The version of python to use for bindings")
    set(PYTHON_ABIFLAGS ""
      CACHE STRING "The ABI flags for the version of Python being used")

  include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

endif(KWIVER_ENABLE_PYTHON)

if (NOT IS_ABSOLUTE "${CMAKE_INSTALL_PREFIX}")
  message(FATAL_ERROR "KWIVER does not support a relative install prefix (${CMAKE_INSTALL_PREFIX}).")
endif ()

# ==================================================================

set( LIB_SUFFIX "" CACHE STRING
  "Library directory suffix. e.g. suffix=\"kwiver\" will install libraries in \"libkwiver\" rather than \"lib\"")
mark_as_advanced( LIB_SUFFIX )

###
# Configure setup script.
configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.sh.in
  ${KWIVER_SETUP_SCRIPT_FILE}
  )

# install set up script
option( KWIVER_INSTALL_SET_UP_SCRIPT "Whether or not to install setup_KWIVER.sh" ON )
mark_as_advanced( KWIVER_INSTALL_SET_UP_SCRIPT )

if( KWIVER_INSTALL_SET_UP_SCRIPT )
  install( PROGRAMS   ${KWIVER_SETUP_SCRIPT_FILE}
    DESTINATION ${CMAKE_INSTALL_PREFIX} )
endif()

###
# Install the basic logger properties file.
file( COPY log4cxx.properties
  DESTINATION  "${KWIVER_BINARY_DIR}" )

install( FILES log4cxx.properties
  DESTINATION ${CMAKE_INSTALL_PREFIX} )

if ( KWIVER_ENABLE_LOG4CXX )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export VITAL_LOGGER_FACTORY=$this_dir/lib/modules/vital_log4cxx_logger.so\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LOG4CXX_CONFIGURATION=$this_dir/log4cxx.properties\n" )
endif()

if (KWIVER_ENABLE_PYTHON)
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PYTHONPATH=$this_dir/${python_site_packages}:$PYTHONPATH\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export SPROKIT_PYTHON_MODULES=kwiver.processes\n" )
endif()

if ( KWIVER_ENABLE_MATLAB )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LD_LIBRARY_PATH=${Matlab_LIBRARY_DIR}:$LD_LIBRARY_PATH\n" )
endif()

if ( fletch_ROOT )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LD_LIBRARY_PATH=${fletch_ROOT}/lib:$LD_LIBRARY_PATH\n" )
endif()

if (KWIVER_ENABLE_TRACK_ORACLE)
  add_subdirectory( track_oracle )
endif()

if (KWIVER_ENABLE_ARROWS)
  add_subdirectory( arrows )
endif()

if (KWIVER_ENABLE_SPROKIT)
  add_subdirectory(sprokit)
endif()

if (KWIVER_ENABLE_EXTRAS)
  add_subdirectory(extras)
endif()

###
# include as the last subdir
add_subdirectory( vital )

##############################################################
# Marking source root for documentation generation
#
kwiver_create_doxygen( kwiver "${CMAKE_CURRENT_LIST_DIR}" )


##################################################################
# Top level installation
#
set(kwiver_cmake_install_dir lib${LIB_SUFFIX}/cmake/kwiver)

# Install rules for CMake utilities
include( kwiver-install-utils )

# Prepare space-separated list of library names for config
get_property(kwiver_libs GLOBAL PROPERTY kwiver_libraries)
string(REPLACE ";" " " kwiver_libs "${kwiver_libs}")

kwiver_configure_file(kwiver-config
  "${KWIVER_SOURCE_DIR}/CMake/kwiver-config-build.cmake.in"
  "${KWIVER_CONFIG_FILE}"
  KWIVER_SOURCE_DIR
  KWIVER_BINARY_DIR
  KWIVER_ENABLE_PYTHON
  BUILD_SHARED_LIBS
  KWIVER_CMAKE_DIR
  kwiver_libs
  fletch_DIR
  EIGEN3_INCLUDE_DIR
  )

kwiver_export_targets("${KWIVER_BINARY_DIR}/kwiver-config-targets.cmake")

# Configure install-tree CMake config file and export associated targets file
set(KWIVER_CONFIG_INSTALL_FILE "${KWIVER_BINARY_DIR}/kwiver-config-install.cmake")
kwiver_configure_file(kwiver-install-config
  "${KWIVER_SOURCE_DIR}/CMake/kwiver-config-install.cmake.in"
  "${KWIVER_CONFIG_INSTALL_FILE}"
  KWIVER_SOURCE_DIR
  KWIVER_BINARY_DIR
  KWIVER_ENABLE_PYTHON
  BUILD_SHARED_LIBS
  CMAKE_INSTALL_PREFIX
  kwiver_libs
  fletch_DIR
  EIGEN3_INCLUDE_DIR
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
