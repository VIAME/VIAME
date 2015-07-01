#
# Top level build for VITAL
#
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fvisibility=hidden KWIVER_HAVE_GCC_VISIBILITY)

include( GenerateExportHeader )

set ( VITAL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
set ( VITAL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} )

###
# Add the CMake directory for CMake modules
#
list(INSERT CMAKE_MODULE_PATH 0
  "${KWIVER_CORE_SOURCE_DIR}/CMake"
  )

# ==================================================================

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

###
#
set( kwiver_public_headers
  camera.h
  camera_intrinsics.h
  camera_io.h
  camera_map.h
  camera_map_io.h
  covariance.h
  descriptor.h
  descriptor_set.h
  eigen_io.h
  exceptions.h
  feature.h
  feature_set.h
  geo_lat_lon.h
#  geo_reference_points_io.h
  homography_f2f.h
  homography_f2w.h
  homography.h
  image_container.h
  image.h
  ins_data.h
  ins_data_io.h
  landmark.h
  landmark_map.h
  landmark_map_io.h
#   local_geo_cs.h
  match_set.h
  matrix.h
  metrics.h
  projected_track_set.h
  rotation.h
  similarity.h
  timestamp.h
  track.h
  track_set.h
  track_set_io.h
  transform.h
  vector.h
  vital-config.h

  exceptions/algorithm.h
  exceptions/base.h
  exceptions/image.h
  exceptions/io.h
  exceptions/math.h
)

set( kwiver_sources
  camera.cxx
  camera_intrinsics.cxx
  camera_io.cxx
  camera_map_io.cxx
  feature.cxx
  geo_lat_lon.cxx
# geo_reference_points_io.cxx
  homography.cxx
  homography_f2f.cxx
  homography_f2w.cxx
  image.cxx
  ins_data.cxx
  ins_data_io.cxx
  landmark.cxx
  landmark_map_io.cxx
#   local_geo_cs.cxx
  metrics.cxx
  projected_track_set.cxx
  rotation.cxx
  similarity.cxx
  timestamp.cxx
  track.cxx
  track_set.cxx
  track_set_io.cxx
  transform.cxx

  exceptions/algorithm.cxx
  exceptions/base.cxx
  exceptions/image.cxx
  exceptions/io.cxx
  exceptions/math.cxx

  # kwiver.cxx # really polygon IO, belongs in kwiver_util
)

kwiver_install_headers(
  ${kwiver_public_headers}
  )

kwiver_add_library( vital SHARED
  ${kwiver_public_headers}
  ${kwiver_sources}
  )

target_link_libraries( vital
  PRIVATE       ${Boost_LIBRARIES} # be more specific
                kwiver_config
                kwiver_logger
  )


add_subdirectory( bindings )

###
# Add tests if enabled
if (KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()

###
# Make appropriate export header file
generate_export_header( vital )
