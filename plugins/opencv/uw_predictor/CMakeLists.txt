###
# Loadable algorithms

find_package( OpenCV REQUIRED )

set( plugin_headers
  classHierarchy.h
#  highgui.h
#  util.h
#  cv.h
  uw_predictor_classifier.h
  SpeciesIDLib.h
  math_common.hpp
  FGExtraction.h
  FGObject.h
  featureExtraction.h
  parameters.h
  snprintf.h
  )

set( plugin_sources
  uw_predictor_classifier.cxx
  classHierarchy.cpp
  FGObject.cpp
  SpeciesIDLib.cpp
  featureExtraction.cpp
  FGExtraction.cpp
  parameters.cpp
  util.cpp
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_uw_predictor_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_uw_predictor
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_uw_predictor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_uw_predictor
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::kwiver_algo_ocv
                       opencv_ml
                       opencv_objdetect
  )

set_target_properties( viame_uw_predictor PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_uw_predictor
  register_algorithms.cxx
  )

target_link_libraries( viame_uw_predictor_plugin
  PUBLIC               kwiver::vital_vpm
  )
