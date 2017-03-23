###
# top level build file for NOAA VIAME Package Util
#

cmake_minimum_required( VERSION 3.3 )

project( VIAME-PACKAGE-BUILDER )

set( CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH} )
include( ExternalProject )

if( NOT DEFINED VIAME_BUILD_INSTALL_PREFIX )
  set( VIAME_BUILD_INSTALL_PREFIX "${VIAME_BINARY_DIR}/install" )
endif()

file( GLOB VIAME_BIN_FILES "${VIAME_BUILD_INSTALL_PREFIX}/bin/*"  )
list( APPEND FIXUP_DIRS ${VIAME_BUILD_INSTALL_PREFIX}/bin )

file( GLOB VIAME_LIB_FILES "${VIAME_BUILD_INSTALL_PREFIX}/lib/*"  )
list( APPEND FIXUP_DIRS ${VIAME_BUILD_INSTALL_PREFIX}/lib )

file( GLOB VIAME_INCLUDE_FILES "${VIAME_BUILD_INSTALL_PREFIX}/include/*"  )
list( APPEND FIXUP_DIRS ${VIAME_BUILD_INSTALL_PREFIX}/include )

file( GLOB VIAME_GUI_PLUGINS_FILES "${VIAME_BUILD_INSTALL_PREFIX}/plugins/*"  )
list( APPEND FIXUP_DIRS ${VIAME_BUILD_INSTALL_PREFIX}/plugins )

file( GLOB VIAME_SPRKIT_PLUGINS_FILES "${VIAME_BUILD_INSTALL_PREFIX}/lib/sprokit/*" )
list( APPEND FIXUP_DIRS ${VIAME_BUILD_INSTALL_PREFIX}/lib/sprokit )

file( GLOB VIAME_VIAME_PLUGINS_FILES "${VIAME_BUILD_INSTALL_PREFIX}/lib/modules/*" )
list( APPEND FIXUP_DIRS ${VIAME_BUILD_INSTALL_PREFIX}/lib/modules )

file( GLOB VIAME_PLUGIN_PLUGINS_FILES "${VIAME_BUILD_INSTALL_PREFIX}/lib/modules/plugin_explorer/*" )
list( APPEND FIXUP_DIRS ${VIAME_BUILD_INSTALL_PREFIX}/lib/modules/plugin_explorer/ )

set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "VIAME" )
set( CPACK_PACKAGE_VENDOR              "Kitware, NOAA, and Friends" )
set( CPACK_PACKAGE_DESCRIPTION_FILE    "${CMAKE_CURRENT_SOURCE_DIR}/README.md" )
set( CPACK_RESOURCE_FILE_LICENSE       "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt" )
set( CPACK_PACKAGE_VERSION_MAJOR       "0" )
set( CPACK_PACKAGE_VERSION_MINOR       "2" )
set( CPACK_PACKAGE_VERSION_PATCH       "1" )
set( CPACK_PACKAGE_INSTALL_DIRECTORY   "VIAME-${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}" )

include( BundleUtilities )
include( InstallRequiredSystemLibraries )

fixup_bundle( "${VIAME_BUILD_INSTALL_PREFIX}/bin/pipeline_runner.exe"
              ""
              "${FIXUP_DIRS}" )

#    if( WIN32 AND NOT UNIX )
#      set( CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
#      set( CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
#      set( CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} My Famous Project")
#      set( CPACK_NSIS_HELP_LINK "http:\\\\\\\\www.my-project-home-page.org")
#      set( CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\www.my-personal-home-page.com")
#      set( CPACK_NSIS_CONTACT "me@my-personal-home-page.com")
#      set( CPACK_NSIS_MODIFY_PATH ON)
#   else( WIN32 AND NOT UNIX )
#      set( CPACK_STRIP_FILES "bin/MyExecutable" )
#      set( CPACK_SOURCE_STRIP_FILES "" )
#   endif()

#SET( CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable" )
include( CPack )
