# C interface library

set(vital_c_public_headers
  algorithm.h

  common.h
  config_block.h
  error_handle.h

  algo_def/bundle_adjust.h
  algo_def/convert_image.h
  algo_def/estimate_canonical_transform.h
  algo_def/estimate_similarity_transform.h
  algo_def/geo_map.h
  algo_def/image_io.h
  algo_def/initialize_cameras_landmarks.h
  algo_def/track_features.h
  algo_def/triangulate_landmarks.h

  types/bounding_box.h
  types/camera.h
  types/camera_intrinsics.h
  types/camera_map.h
  types/color.h
  types/covariance.h
  types/descriptor.h
  types/descriptor_set.h
  types/descriptor_set.hxx
  types/detected_object.h
  types/detected_object_type.h
  types/detected_object_set.h
  types/detected_object_set.hxx
  types/eigen.h
  types/feature.h
  types/feature_track_set.h
  types/homography.h
  types/image.h
  types/image_container.h
  types/image_container.hxx
  types/landmark.h
  types/landmark_map.h
  types/object_track_set.h
  types/rotation.h
  types/similarity.h
  types/track.h
  types/track_set.h
  types/track_set.hxx
  )

set(vital_c_private_headers
  # Created by CMake during configuration
  ${CMAKE_CURRENT_BINARY_DIR}/vital_c_export.h
  helpers/algorithm.h
  helpers/c_utils.h
  helpers/camera.h
  helpers/camera_intrinsics.h
  helpers/camera_map.h
  helpers/config_block.h
  helpers/descriptor.h
  helpers/descriptor_set.h
  helpers/detected_object.h
  helpers/detected_object_type.h
  helpers/detected_object_set.h
  helpers/feature.h
  helpers/image_container.h
  helpers/landmark.h
  helpers/landmark_map.h
  helpers/track.h
  helpers/track_set.h
  )

set(vital_c_sources
  algorithm.cxx
  common.cxx
  config_block.cxx
  error_handle.cxx

  algo_def/bundle_adjust.cxx
  algo_def/convert_image.cxx
  algo_def/estimate_canonical_transform.cxx
  algo_def/estimate_similarity_transform.cxx
  algo_def/geo_map.cxx
  algo_def/image_io.cxx
  algo_def/initialize_cameras_landmarks.cxx
  algo_def/track_features.cxx
  algo_def/triangulate_landmarks.cxx

  helpers/c_utils.cxx

  types/bounding_box.cxx
  types/camera.cxx
  types/camera_intrinsics.cxx
  types/camera_map.cxx
  types/color.cxx
  types/covariance.cxx
  types/descriptor.cxx
  types/descriptor_set.cxx
  types/detected_object.cxx
  types/detected_object_type.cxx
  types/detected_object_set.cxx
  types/eigen.cxx
  types/feature.cxx
  types/feature_track_set.cxx
  types/homography.cxx
  types/image.cxx
  types/image_container.cxx
  types/landmark.cxx
  types/landmark_map.cxx
  types/object_track_set.cxx
  types/rotation.cxx
  types/similarity.cxx
  types/track.cxx
  types/track_set.cxx
  )

kwiver_install_headers(
  ${vital_c_public_headers}
  SUBDIR     vital/bindings/c
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_c_export.h
  SUBDIR     vital/bindings/c
  NOPATH
  )

kwiver_add_library( vital_c
  ${vital_c_public_headers}
  ${vital_c_private_headers}
  ${vital_c_sources}
  )

target_link_libraries( vital_c
  PRIVATE     vital
              vital_vpm
              vital_algo
  )
