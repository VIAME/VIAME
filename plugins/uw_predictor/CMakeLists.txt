###
# Loadable algorithms

find_package( OpenCV REQUIRED )

set( plugin_headers
  classHierarchy.h
#  highgui.h
  register_algorithms.h
#  util.h
#  cv.h
  uw_predictor_classifier.h
  SpeciesIDLib.h
  math_common.hpp
  FGExtraction.h
  FGObject.h
  featureExtraction.h
  parameters.h
  )

set( plugin_sources
  algorithm_plugin_interface.cxx
  register_algorithms.cxx
  uw_predictor_classifier.cxx

  classHierarchy.cpp
  FGObject.cpp
  SpeciesIDLib.cpp
  featureExtraction.cpp
  FGExtraction.cpp
  parameters.cpp
  util.cpp

  register_algorithms.h    register_algorithms.cxx
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

target_link_libraries( viame_uw_predictor
  PUBLIC               vital
                       kwiver_algo_ocv
  )

algorithms_create_plugin( viame_uw_predictor
  algorithm_plugin_interface.cxx
  )
