
###
# Sub folder options
##
if( VIAME_ENABLE_UW_PREDICTOR )
  add_subdirectory( uw_predictor )
endif()

###
# Loadable algorithms
##

find_package( OpenCV REQUIRED )

include_directories( "${OpenCV_INCLUDE_DIRS}" )
link_directories( "${OpenCV_LIBRARY_DIR}" )

set( plugin_headers
  ocv_image_enhancement.h
  ocv_debayer_filter.h
  ocv_random_hue_shift.h
  ocv_stereo_depth_map.h
  ocv_rectified_stereo_disparity_map.h
  ocv_target_detector.h
  ocv_optimize_stereo_cameras.h
  ocv_stereo_feature_track_filter.h
  ocv_kmedians.h
  split_image_habcam.h  
  )

set( plugin_sources
  ocv_image_enhancement.cxx
  ocv_debayer_filter.cxx
  ocv_random_hue_shift.cxx
  ocv_stereo_depth_map.cxx
  ocv_rectified_stereo_disparity_map.cxx
  ocv_target_detector.cxx
  ocv_optimize_stereo_cameras.cxx
  ocv_stereo_feature_track_filter.cxx
  ocv_kmedians.cxx
  split_image_habcam.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_opencv_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_opencv
  ${plugin_headers}
  ${plugin_sources}
  )

get_target_property( results viame_opencv COMPILE_DEFINITIONS )

if( "${OpenCV_VERSION}" VERSION_LESS "3.0.0" )
  target_compile_definitions( viame_opencv PRIVATE -DVIAME_OPENCV_VER_2 )
endif()

get_target_property( results viame_opencv COMPILE_DEFINITIONS )

target_link_libraries( viame_opencv
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       kwiver::kwiversys
                       kwiver::kwiver_algo_ocv
                       ${OpenCV_LIBRARIES}
  )

target_include_directories(viame_opencv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR})

set_target_properties( viame_opencv PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_opencv
  register_algorithms.cxx
  )

target_link_libraries( viame_opencv_plugin
  PUBLIC               kwiver::vital_vpm
  )

# Eigen is required for MVG triangulation
find_package( Eigen3 REQUIRED NO_MODULE )

# Add process library
set( process_headers
  ocv_stereo_utils.h
  ocv_measurement_process.h
  ocv_add_keypoints_from_poly.h
)

set( process_sources
  register_processes.cxx
  ocv_stereo_utils.cxx
  ocv_measurement_process.cxx
  ocv_add_keypoints_from_poly.cxx
)

kwiver_add_plugin( viame_processes_opencv
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
  PRIVATE          kwiver::vital
                   kwiver::vital_algo
                   kwiver::vital_config
                   kwiver::vital_logger
                   kwiver::vital_util
                   kwiver::vital_vpm
                   kwiver::sprokit_pipeline
                   kwiver::kwiver_algo_ocv
                   kwiver::kwiver_algo_mvg
                   Eigen3::Eigen
                   ${OpenCV_LIBRARIES}
                   viame_opencv
)

if( VIAME_ENABLE_PYTHON )
  set( _rel_modnames
    __init__
    ocv_multimodal_registration
    ocv_fft_filter_based_on_ref
    ocv_stereo_algos
    ocv_stereo_demo
    ocv_stereo_expt
    ocv_stereo_pipeline
    ocv_stereo_processes
    ocv_stereo_utils
    )

  foreach( rel_modname IN ITEMS ${_rel_modnames} )
    string( REPLACE "." "/" rel_modpath_noext ${rel_modname} )
    kwiver_add_python_module(
      "${CMAKE_CURRENT_SOURCE_DIR}/${rel_modpath_noext}.py"
      "processes/opencv"
      "${rel_modname}" )
  endforeach()
endif()

if (VIAME_BUILD_TESTS)
  add_subdirectory(tests)
endif()