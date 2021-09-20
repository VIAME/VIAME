
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
  if( NOT EXISTS "${VIAME_DIR}/setup_viame.sh" AND NOT EXISTS "${VIAME_DIR}/setup_viame.bat" )
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
  example_filter_process.py
  )

set( PLUGIN_PATH "viame/processes/external_example" )

if( EXISTS "${VIAME_DIR}/lib/python3.6/site-packages" )
  set( PYTHON_PATH "lib/python3.6/site-packages" )
else()
  set( PYTHON_PATH "lib/python3.6/dist-packages" )
endif()

foreach( PLUGIN_FILE ${PLUGIN_FILES} )
  install( FILES "${PLUGIN_FILE}"
    DESTINATION "${VIAME_DIR}/${PYTHON_PATH}/${PLUGIN_PATH}"
    )
endforeach()
