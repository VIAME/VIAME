#ckwg +4
# Copyright 2010,2014 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

# Locate the system installed TinyXML
# The following variables will be set:
#
# TinyXML_FOUND       - Set to true of the system TinyXML can be found
# TinyXML_INCLUDE_DIR - The path to the tinyxml header files
# TinyXML_LIBRARY     - The full path to the TinyXML library

if( TinyXML_DIR )
    find_package( TinyXML ${TinyXML_FIND_VERSION} NO_MODULE )
elseif( NOT TinyXML_FOUND )
  include(CommonFindMacros)

  setup_find_root_context(TinyXML)
  find_path( TinyXML_INCLUDE_DIR tinyxml.h ${TinyXML_FIND_OPTS})
  find_library( TinyXML_LIBRARY tinyxml ${TinyXML_FIND_OPTS})
  restore_find_root_context(TinyXML)

  include( FindPackageHandleStandardArgs )
  FIND_PACKAGE_HANDLE_STANDARD_ARGS( TinyXML TinyXML_INCLUDE_DIR TinyXML_LIBRARY )

  if( TINYXML_FOUND )
    # Check to see if TinyXML was built with STL support or not
    include( CheckCXXSourceCompiles )
    set( CMAKE_REQUIRED_DEFINITIONS "-DTIXML_USE_STL" )
    set( CMAKE_REQUIRED_INCLUDES ${TinyXML_INCLUDE_DIR} )
    set( CMAKE_REQUIRED_LIBRARIES ${TinyXML_LIBRARY})

    #The following approach, while unfortunate, is required under certain circumstances.
    #CMake always does a try/compile in Debug mode. In Visual Studio >= 2010 we can't
    #link against the tinyxml.lib when it's Release, nor can we always decide at CMake config time
    #which mode we will run in. Running both modes explicitly and testing whether either succeeds
    #will tell us what we need to know
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Debug")
    CHECK_CXX_SOURCE_COMPILES("
      #include <tinyxml.h>
      int main() { TiXmlNode *node; std::cin >> *node; } "
      TinyXML_USE_STL_D
      )
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
    CHECK_CXX_SOURCE_COMPILES("
      #include <tinyxml.h>
      int main() { TiXmlNode *node; std::cin >> *node; } "
      TinyXML_USE_STL_R
      )

    if( TinyXML_USE_STL_D OR TinyXML_USE_STL_R)
      add_definitions( -DTIXML_USE_STL )
    endif()

    # Determine the TinyXML version found
    file( READ ${TinyXML_INCLUDE_DIR}/tinyxml.h TinyXML_INCLUDE_FILE )
    string( REGEX REPLACE
      ".*TIXML_MAJOR_VERSION = ([0-9]+).*" "\\1"
      TinyXML_VERSION_MAJOR "${TinyXML_INCLUDE_FILE}" )
    string( REGEX REPLACE
      ".*TIXML_MINOR_VERSION = ([0-9]+).*" "\\1"
      TinyXML_VERSION_MINOR "${TinyXML_INCLUDE_FILE}" )
    string( REGEX REPLACE
      ".*TIXML_PATCH_VERSION = ([0-9]+).*" "\\1"
      TinyXML_VERSION_PATCH "${TinyXML_INCLUDE_FILE}" )
    set( TinyXML_VERSION "${TinyXML_VERSION_MAJOR}.${TinyXML_VERSION_MINOR}.${TinyXML_VERSION_PATCH}" )

    # Determine version compatibility
    if( TinyXML_FIND_VERSION )
      if( TinyXML_FIND_VERSION VERSION_EQUAL TinyXML_VERSION )
        message( STATUS "TinyXML version: ${TinyXML_VERSION}" )
        set( TinyXML_FOUND TRUE )
      else()
        if( (TinyXML_FIND_VERSION_MAJOR EQUAL TinyXML_VERSION_MAJOR) AND
            (TinyXML_FIND_VERSION_MINOR EQUAL TinyXML_VERSION_MINOR) AND
            (TinyXML_FIND_VERSION VERSION_LESS TinyXML_VERSION) )
          message( STATUS "TinyXML version: ${TinyXML_VERSION}" )
          set( TinyXML_FOUND TRUE )
        endif()
      endif()
    else()
      message( STATUS "TinyXML version: ${TinyXML_VERSION}" )
      set( TinyXML_FOUND TRUE )
    endif()

    unset( TINYXML_FOUND )
  endif()
endif()
