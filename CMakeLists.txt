###
# Top level build script for kwiver-build
#
cmake_minimum_required(VERSION 3.3)

project(KWIVER)

###
# KWIVER version
set(KWIVER_VERSION_MAJOR 1)
set(KWIVER_VERSION_MINOR 2)
set(KWIVER_VERSION_PATCH 0)
set(KWIVER_VERSION "${KWIVER_VERSION_MAJOR}.${KWIVER_VERSION_MINOR}.${KWIVER_VERSION_PATCH}")

###
# Look for and use Fletch to find dependencies
#
set( fletch_DIR "" CACHE PATH "Path to FLETCH packages" )
if ( IS_DIRECTORY ${fletch_DIR} )
  find_package( fletch NO_MODULE )
  if ( fletch_FOUND )
    # Search for libraries in the fletch install root
    list(APPEND CMAKE_PREFIX_PATH "${fletch_ROOT}")
  endif()
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
# Create initial setup shell script
set(KWIVER_SETUP_SCRIPT_FILE    "${KWIVER_BINARY_DIR}/setup_KWIVER.sh" )
# Create initial setup batch script
set(KWIVER_SETUP_BATCH_FILE    "${KWIVER_BINARY_DIR}/setup_KWIVER.bat" )
if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
 if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 19.10)
    set(_vcVersion vc15)
  elseif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 19)
    set(_vcVersion vc14)
  else()
    message(FATAL_ERROR "KWIVER requires Visual Studio 2015 or greater")
  endif()
endif()

# prepend our CMake resources
list(INSERT CMAKE_MODULE_PATH 0 ${KWIVER_CMAKE_DIR})

# import CMake future modules
include(kwiver-cmake-future)

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
include(CMakeDependentOption)

OPTION(KWIVER_BUILD_SHARED     "Build KWIVER components shared or not" TRUE )
MARK_AS_ADVANCED( KWIVER_BUILD_SHARED )

OPTION(KWIVER_ENABLE_TESTS     "Enable testing" OFF )
OPTION(KWIVER_ENABLE_ARROWS    "Enable building arrows" OFF )
OPTION(KWIVER_ENABLE_TOOLS     "Enable tools" OFF )
OPTION(KWIVER_ENABLE_TRACK_ORACLE  "Enable track_oracle" OFF )
OPTION(KWIVER_ENABLE_EXAMPLES  "Enable examples" OFF )

# if sprokit enabled
OPTION(KWIVER_ENABLE_SPROKIT "Enable building sprokit" OFF )

CMAKE_DEPENDENT_OPTION(KWIVER_ENABLE_PROCESSES "Should the KWIVER Sprokit Processes Be Built?" OFF
  "KWIVER_ENABLE_ARROWS AND KWIVER_ENABLE_SPROKIT" OFF )

OPTION(KWIVER_USE_BUILD_TREE "Whether to include the build-tree plugin directory to the default plugin path list" OFF )
MARK_AS_ADVANCED(KWIVER_USE_BUILD_TREE)

option( KWIVER_ENABLE_EXTRAS "Enable extras content" OFF )

## TODO: How should pytgon and c lib interact?
OPTION( KWIVER_ENABLE_C_BINDINGS     "Enable C bindings libraries" OFF)
CMAKE_DEPENDENT_OPTION( KWIVER_ENABLE_PYTHON    "Enable python code" OFF
  KWIVER_ENABLE_C_BINDINGS OFF )

if (KWIVER_ENABLE_PYTHON)

  if (NOT WIN32)
    OPTION(KWIVER_SYMLINK_PYTHON "Symlink python files instead of copying. Allows for more dynamic python development." Off)
  endif()

  if ( NOT KWIVER_ENABLE_C_BINDINGS )
    message( SEND_ERROR  "Python can not be enabled unless KWIVER_ENABLE_C_BINDINGS is also enabled." )
  endif()
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

if (NOT IS_ABSOLUTE "${CMAKE_INSTALL_PREFIX}")
  message(FATAL_ERROR "KWIVER does not support a relative install prefix (${CMAKE_INSTALL_PREFIX}).")
endif ()

# ==================================================================

set( LIB_SUFFIX "" CACHE STRING
  "Library directory suffix. e.g. suffix=\"kwiver\" will install libraries in \"libkwiver\" rather than \"lib\"")
mark_as_advanced( LIB_SUFFIX )

###
# Configure setup script.
set(LIBRARY_PATH_VAR "LD_LIBRARY_PATH")
if( APPLE )
  set(LIBRARY_PATH_VAR "DYLD_FALLBACK_LIBRARY_PATH")
endif()

configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.sh.in
  ${KWIVER_SETUP_SCRIPT_FILE}
  @ONLY
  )

if(fletch_BUILT_WITH_CUDA)
  list(APPEND EXTRA_WIN32_PATH "${CUDA_TOOLKIT_ROOT_DIR}/bin")
endif()
if(fletch_BUILT_WITH_CUDNN)
  list(APPEND EXTRA_WIN32_PATH "${CUDNN_TOOLKIT_ROOT_DIR}/bin")
endif()
configure_file(
  ${KWIVER_CMAKE_DIR}/setup_KWIVER.bat.in
  ${KWIVER_SETUP_BATCH_FILE}
  @ONLY
  )


# install set up script
option( KWIVER_INSTALL_SET_UP_SCRIPT "Creates a setup_KWIVER script (.sh and .bat) that will add properly add kwiver to a shell/cmd prompt" ON )
mark_as_advanced( KWIVER_INSTALL_SET_UP_SCRIPT )

if( KWIVER_INSTALL_SET_UP_SCRIPT )
  install( PROGRAMS   ${KWIVER_SETUP_SCRIPT_FILE}
    DESTINATION ${CMAKE_INSTALL_PREFIX} )
  if(WIN32)
    install( PROGRAMS   ${KWIVER_SETUP_BATCH_FILE}
      DESTINATION ${CMAKE_INSTALL_PREFIX} )
  endif()
endif()

if ( fletch_FOUND )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=%PATH%;${fletch_ROOT}/bin\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=%PATH%;${fletch_ROOT}/x64/${_vcVersion}/bin\n" )

  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export ${LIBRARY_PATH_VAR}=${fletch_ROOT}/lib:$${LIBRARY_PATH_VAR}\n" )
endif()

###
# Install the basic logger properties file.
file( COPY log4cxx.properties       DESTINATION  "${KWIVER_BINARY_DIR}" )
install( FILES log4cxx.properties   DESTINATION ${CMAKE_INSTALL_PREFIX} )

if ( KWIVER_ENABLE_LOG4CXX )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export VITAL_LOGGER_FACTORY=$this_dir/lib/modules/vital_log4cxx_logger\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LOG4CXX_CONFIGURATION=$this_dir/log4cxx.properties\n" )
  if(WIN32)
    message(STATUS "Log4CXX is not supported on windows, if no other logger is provided, the default will be used")
  endif()

endif()

###
file( COPY log4cplus.properties       DESTINATION  "${KWIVER_BINARY_DIR}" )
install( FILES log4cplus.properties   DESTINATION ${CMAKE_INSTALL_PREFIX} )

if ( KWIVER_ENABLE_LOG4CPLUS )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export VITAL_LOGGER_FACTORY=$this_dir/lib/modules/vital_log4cplus_logger\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LOG4CPLUS_CONFIGURATION=$this_dir/log4cplus.properties\n" )

  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set VITAL_LOGGER_FACTORY=vital_log4cplus_logger\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set LOG4CPLUS_CONFIGURATION=%~dp0/log4cplus.properties\n" )
endif()


if (KWIVER_ENABLE_PYTHON)
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "# Python environment\n")
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PYTHON_LIBRARY=\"${PYTHON_LIBRARY}\"\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export PYTHONPATH=$this_dir/${python_site_packages}:$PYTHONPATH\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "\n# additional python mudules to load, separated by ':'\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export SPROKIT_PYTHON_MODULES=kwiver.processes\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "\n# set to suppress loading python modules/processes\n" )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "# export SPROKIT_NO_PYTHON_MODULES\n\n" )

  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "# Python environment\n")
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PYTHON_LIBRARY=${PYTHON_LIBRARY}\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PYTHONPATH=%~dp0/lib/%config%/python2.7/site-packages\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "\n:: additional python mudules to load, separated by ':'\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set SPROKIT_PYTHON_MODULES=kwiver.processes\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "\n:: set to suppress loading python modules/processes\n" )
  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "::set SPROKIT_NO_PYTHON_MODULES=false\n\n" )
endif()

if ( KWIVER_ENABLE_MATLAB )
  file( APPEND "${KWIVER_SETUP_SCRIPT_FILE}" "export LD_LIBRARY_PATH=${Matlab_LIBRARY_DIR}:$LD_LIBRARY_PATH\n" )

  file( APPEND "${KWIVER_SETUP_BATCH_FILE}" "set PATH=%PATH%;${Matlab_LIBRARY_DIR}\n" )
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

if (KWIVER_ENABLE_EXAMPLES)
  add_subdirectory(examples)
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

# setup general paths to includes and libs for packages we use/provide
set( KWIVER_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
set( KWIVER_LIBRARY_DIRS "")

if ( NOT fletch_ENABLED_Boost)
  list( APPEND KWIVER_INCLUDE_DIRS "${BOOST_ROOT_ARG}/include")
  list( APPEND KWIVER_LIBRARY_DIRS "${BOOST_ROOT_ARG}/lib")
endif()

if (fletch_FOUND)
  # since we have fletch, add include and lib dirs to export
  list( APPEND KWIVER_INCLUDE_DIRS "${fletch_ROOT}/include")
  list( APPEND KWIVER_LIBRARY_DIRS "${fletch_ROOT}/lib")
endif()

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
  KWIVER_INCLUDE_DIRS
  KWIVER_LIBRARY_DIRS
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
  KWIVER_INCLUDE_DIRS
  KWIVER_LIBRARY_DIRS
  LIB_SUFFIX
  KWIVER_VERSION
  )

kwiver_install(
  FILES       "${KWIVER_CONFIG_INSTALL_FILE}"
  DESTINATION "${kwiver_cmake_install_dir}"
  RENAME      kwiver-config.cmake
  )

kwiver_install(
  EXPORT      ${kwiver_export_name}
  NAMESPACE   kwiver::
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
