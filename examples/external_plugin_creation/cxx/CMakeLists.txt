
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

find_package( KWIVER REQUIRED HINTS "${VIAME_DIR}/lib/cmake/kwiver" )
find_package( OpenCV REQUIRED HINTS "${VIAME_DIR}" )

# Set default paths
set( CMAKE_INSTALL_PREFIX "${VIAME_DIR}" CACHE PATH "Default install dir" FORCE )
set( PLUGIN_NAME "example_plugin" )

# Include VIAME directories
include_directories( "${VIAME_DIR}/include" )
include_directories( "${VIAME_DIR}/include/eigen3" )

link_directories( "${VIAME_DIR}/lib" )

# Point to plugin source files and headers
set( plugin_headers
  example_detector.h
  )

set( plugin_sources
  example_detector.cxx
  register_algorithms.cxx
  )

# Compile plugin lshared library
add_library( ${PLUGIN_NAME}
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( ${PLUGIN_NAME}
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::kwiver_algo_ocv
  )

set_target_properties( ${PLUGIN_NAME}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/modules"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/modules"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )

set_target_properties( ${PLUGIN_NAME}
  PROPERTIES
  PREFIX "" )

install( TARGETS ${PLUGIN_NAME}
  ARCHIVE DESTINATION lib/modules
  LIBRARY DESTINATION lib/modules
  RUNTIME DESTINATION bin
  )
