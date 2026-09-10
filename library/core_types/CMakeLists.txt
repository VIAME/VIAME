#
# vital/types library and headers
#
# This sub-library is intended to be home to canonical typedefs and other
# commonly useful type definitions.
#

set( vital_types_public_headers
  camera.h
  camera_map.h
  camera_perspective_map.h
  class_map.h
  color.h
  covariance.h
  feature_set.h
  image_container.h
  image_container_set.h
  iterator.h
  landmark_map.h
  match_set.h
  matrix.h
  mesh_container.h
  metadata_map.h
  metadata_tags.h
  query_result_set.h
  set.h
  signal.h
  timestamp_config.h
  track_descriptor_set.h
  track_interval.h
  transform_2d.h
  vector.h
  )
set( vital_types_classes
  activity
  activity_type
  attribute_set
  bounding_box
  camera_intrinsics
  camera_perspective
  camera_rig
  camera_rpc
  color_space
  context
  category_hierarchy
  database_query
  descriptor
  descriptor_request
  descriptor_set
  detected_object
  detected_object_set
  detected_object_set_util
  detected_object_type
  essential_matrix
  feature
  feature_track_set
  fundamental_matrix
  geodesy
  geo_point
  geo_polygon
  ground_control_point
  homography
  homography_f2f
  image_container_set_simple
  tiled_image_container
  tiled_image_container_simple
  image
  iqr_feedback
  landmark
  local_tangent_space
  mesh
  metadata
  metadata_stream
  metadata_stream_from_map
  metadata_traits
  object_track_set
  pointcloud
  point
  polygon
  query_result
  rotation
  sfm_constraints
  similarity
  timestamp
  track
  track_descriptor
  track_set
  uid
  video_raw_image
  video_raw_metadata
  video_settings
  video_uninterpreted_data
  )

set(vital_types_sources
)

foreach(class IN LISTS vital_types_classes)
  list(APPEND vital_types_public_headers "${class}.h")
  list(APPEND vital_types_sources "${class}.cxx")
endforeach()
  

kwiver_install_headers(  # generated export header file
  ${CMAKE_CURRENT_BINARY_DIR}/vital_types_export.h
  SUBDIR viame/core_types
  NOPATH
)
kwiver_add_library(
  vital_types SHARED
  ${vital_types_public_headers}
  ${vital_types_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/vital_types_export.h
  )

# Depend on file configuration
add_dependencies( vital_types kwiver_configure )

target_link_libraries( vital_types
  PRIVATE
            vital_logger
            vital_util
  PUBLIC    Eigen3::Eigen
            vital_config
  )

# Republished under the prefix the imported sources use.
viame_lite_generated( vital_types_export.h core_types )
