# Build / Install plugin containing Ceres algorithm implementations + support
# structures

set( ceres_headers_public
  bundle_adjust.h
  lens_distortion.h
  optimize_cameras.h
  reprojection_error.h
  camera_intrinsic_prior.h
  camera_position.h
  camera_smoothness.h
  )
set( ceres_headers_private
  options.h
  types.h
  )

kwiver_install_headers(
  SUBDIR     arrows/ceres
  ${ceres_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_ceres_export.h
  NOPATH   SUBDIR     arrows/ceres
  )

set( ceres_sources
  options.cxx
  bundle_adjust.cxx
  optimize_cameras.cxx
  reprojection_error.cxx
  types.cxx
  )

kwiver_add_library( kwiver_algo_ceres
  ${ceres_headers_public}
  ${ceres_headers_private}
  ${ceres_sources}
  )
target_link_libraries( kwiver_algo_ceres
  PUBLIC               kwiver_algo_core
  PRIVATE              ceres
  )
if(NOT CERES_USES_MINIGLOG)
  target_link_libraries( kwiver_algo_ceres
    PRIVATE              glog
    )
endif()

algorithms_create_plugin( kwiver_algo_ceres
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
