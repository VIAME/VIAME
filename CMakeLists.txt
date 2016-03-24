###
#
# Top level CMake file for vital
#

cmake_minimum_required(VERSION 3.0)

project(VITAL)

include(CMakeDependentOption)

# This is a windows only requirement
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${VITAL_BINARY_DIR}/bin")

###
# VITAL version
#
set(VITAL_VERSION_MAJOR 0)
set(VITAL_VERSION_MINOR 1)
set(VITAL_VERSION_PATCH 1)
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

###
# KWIVER CMake config
#
include(kwiver-utils)
include(vital-flags) # vital specific CXX flags

##
# check compiler support
include( kwiver-configcheck )

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

set( LIB_SUFFIX "" CACHE STRING
  "Library directory suffix. e.g. suffix=\"vital\" will install libraries in \"libvital\" rather than \"lib\"")
mark_as_advanced( LIB_SUFFIX )

set(BUILD_SHARED_LIBS TRUE)

# installing system setup script
find_package(PythonInterp)      # needed for PYTHON_VERSION_MAJOR et al
if (fletch_ROOT)
    set(SET_FLETCH_LD_LIBRARY_PATH "export LD_LIBRARY_PATH=${fletch_ROOT}/lib:$LD_LIBRARY_PATH")
    set(SET_FLETCH_PYTHON_PATH "export PYTHONPATH=${fletch_ROOT}/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages:$PYTHONPATH")
endif()

##
# Create initial setup script
set(VITAL_SETUP_SCRIPT_FILE  ${VITAL_BINARY_DIR}/setup_VITAL.sh )

configure_file(
  ${VITAL_CMAKE_DIR}/setup_VITAL.sh.in
  ${VITAL_SETUP_SCRIPT_FILE}
  )

# install set up script
install( PROGRAMS   ${VITAL_SETUP_SCRIPT_FILE}
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
  fletch_DIR
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

##################################################################
# Top level packaging
#
# if log4cxx emabled, require log4cxx RPM


set(CPACK_PACKAGE_NAME              "vital")
set(CPACK_PACKAGE_VENDOR            "Kitware, Inc.")
set(CPACK_PACKAGE_CONTACT           "kitware@kitware.com")
set(CPACK_MONOLITHIC_INSTALL        true)
set(CPACK_PACKAGE_VERSION_MAJOR     "${VITAL_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR     "${VITAL_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH     "${VITAL_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION           "${VITAL_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE     "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_FILE_NAME         "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}")
## set(CPACK_PACKAGE_ICON              "${CMAKE_CURRENT_SOURCE_DIR}/packaging\\\\vital_install.bmp")

# NSIS Package options
## set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\mapgui.exe")
## set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/gui/icons\\\\mapgui.ico")
## set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/gui/icons\\\\mapgui.ico")
set(CPACK_NSIS_CONTACT "${CPACK_PACKAGE_CONTACT}")
set(CPACK_NSIS_HELP_LINK "http://www.kitware.com")
set(CPACK_NSIS_URL_INFO_ABOUT "http://www.kwiver.org")
set(CPACK_NSIS_MENU_LINKS "http://www.kwiver.org" "KWIVER Website")

if(APPLE)
  set(CPACK_BINARY_DRAGNDROP ON)
  set(CPACK_BINARY_PACKAGEMAKER OFF)
  set(CPACK_SET_DESTDIR TRUE)
  ## set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/packaging/vital_drive.icns")
endif()
include (CPack)
