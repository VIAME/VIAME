###
# Evaluation
#
# Scoring detections and tracks against groundtruth -- what `viame score`
# runs. From the `core` plugin in P2-T07. It registers nothing.
##

set( plugin_headers
  evaluate_models.h
  )

set( plugin_sources
  evaluate_models.cxx
  )

viame_install_headers(
  SUBDIR     viame/evaluation
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_evaluation_export.h
  NOPATH   SUBDIR     viame/evaluation
  )

viame_add_library( viame_evaluation
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_evaluation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_evaluation
  PUBLIC               kwiver::vital
                       kwiver::vital_config
                       kwiver::vital_logger
                       kwiver::vital_util
                       # the groundtruth and detection CSV reader
                       viame_file_io
  )

# Evaluation scores sequences in parallel.
find_package( OpenMP )
if( OpenMP_CXX_FOUND )
  target_link_libraries( viame_evaluation PRIVATE OpenMP::OpenMP_CXX )
endif()

set_target_properties( viame_evaluation PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

###
# Tests
##
if( VIAME_ENABLE_TESTS )
  add_subdirectory( tests )
endif()
