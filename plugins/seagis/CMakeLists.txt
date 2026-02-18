# Build / Install Plugin containing SEAGIS measurement implementations

# Process library
set( process_headers
  seagis_measurement_process.h
)

set( process_sources
  register_processes.cxx
  seagis_measurement_process.cxx
)

# Mock mode: compile the mock implementation instead of linking the real library
if( VIAME_BUILD_SEAGIS_TEST_LIB )
  list( APPEND process_sources StereoLibLX_mock.cxx )
  message( STATUS "SEAGIS: Using mock StereoLibLX (no licence required)" )
endif()

# Include SEAGIS headers
include_directories( SYSTEM ${SEAGIS_INCLUDE_DIR} )

set( SEAGIS_LINK_LIBRARIES
  kwiver::vital
  kwiver::vital_algo
  kwiver::vital_config
  kwiver::vital_exceptions
  kwiver::vital_logger
  kwiver::vital_util
  kwiver::sprokit_pipeline
  viame_core
)

if( NOT VIAME_BUILD_SEAGIS_TEST_LIB )
  list( APPEND SEAGIS_LINK_LIBRARIES ${SEAGIS_LIBRARY} )
endif()

# OpenCV for epipolar template matching
if( VIAME_ENABLE_OPENCV )
  find_package( OpenCV REQUIRED )
  include_directories( "${OpenCV_INCLUDE_DIRS}" )
  list( APPEND SEAGIS_LINK_LIBRARIES ${OpenCV_LIBRARIES} kwiver::kwiver_algo_ocv )
endif()

kwiver_add_plugin( viame_processes_seagis
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          ${SEAGIS_LINK_LIBRARIES}
)

target_include_directories( viame_processes_seagis PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../core>
  )
