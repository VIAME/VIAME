# Build / Install plugin containing Ceres algorithm implementations + support
# structures

set( ceres_headers_public
  bundle_adjust.h
  lens_distortion.h
  register_algorithms.h
  reprojection_error.h
  )
set( ceres_headers_private
  types.h
  )
kwiver_install_headers(
  SUBDIR     arrows/algorithms/ceres
  ${ceres_headers_public}
  )
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/algorithms_ceres_export.h
  NOPATH   SUBDIR     arrows/algorithms/ceres
  )

set( ceres_sources
  bundle_adjust.cxx
  register_algorithms.cxx
  types.cxx
  )
kwiver_add_library( algorithms_ceres

  ${ceres_headers_public}
  ${ceres_headers_private}
  ${ceres_sources}
  )
target_link_libraries( algorithms_ceres
  PUBLIC               algorithms_core
                       ceres
  )

algorithms_create_plugin( algorithms_ceres
  algorithm_plugin_interface.cxx
  )
