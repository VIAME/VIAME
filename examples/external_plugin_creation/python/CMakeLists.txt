
project( ExampleExternalDetector )
cmake_minimum_required( VERSION 3.3 )
set( CMAKE_CXX_STANDARD 11 )

# Make sure we know where the VIAME installation rests
set( VIAME_DIR "" CACHE PATH "Location of VIAME installation" )
set( BUILD_SHARED_LIBS ON CACHE BOOL "Build with shared libraries" )

if( VIAME_DIR )
  if( NOT EXISTS "${VIAME_DIR}" )
    message( FATAL_ERROR "VIAME directory does not exist" )
  endif()
  if( NOT EXISTS "${VIAME_DIR}/lib/modules" )
    message( FATAL_ERROR "VIAME directory does not appear to be valid" )
  endif()
else()
  message( FATAL_ERROR "VIAME install directory not set" )
endif()

# Set default paths
set( CMAKE_INSTALL_PREFIX "${VIAME_DIR}" CACHE PATH "Default install dir" FORCE )

# Install python files to correct directory
set( PLUGIN_NAME "example_external_plugin" )

set( PLUGIN_FILES
  example_filter.py
  )

install( FILES example_filter.py
  DESTINATION "${VIAME_DIR}/lib/python2.7/site-packages"
  )
