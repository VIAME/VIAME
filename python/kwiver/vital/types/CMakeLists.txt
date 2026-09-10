set( THIS_MODULE vital/types )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py "${THIS_MODULE}" __init__ )

set( vital_python_headers
     image.h
     image_container.h
  )

set( vital_python_sources
     image.cxx
     image_container.cxx
     types_module.cxx
   )

kwiver_add_python_library(
  types
  "${THIS_MODULE}"
  SOURCES ${vital_python_headers}
          ${vital_python_sources}
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  activity
  "${THIS_MODULE}"
  SOURCES activity.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  activity_type
  "${THIS_MODULE}"
  SOURCES activity_type.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  bounding_box
  "${THIS_MODULE}"
  SOURCES bounding_box.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  camera
  "${THIS_MODULE}"
  SOURCES camera.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  camera_intrinsics
  "${THIS_MODULE}"
  SOURCES camera_intrinsics.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  camera_map
  "${THIS_MODULE}"
  SOURCES camera_map.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  camera_perspective
  "${THIS_MODULE}"
  SOURCES camera_perspective.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  camera_perspective_map
  "${THIS_MODULE}"
  SOURCES camera_perspective_map.cxx
  PRIVATE pybind11::pybind11
           vital
  )

kwiver_add_python_library(
  camera_rpc
  "${THIS_MODULE}"
  SOURCES camera_rpc.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  category_hierarchy
  "${THIS_MODULE}"
  SOURCES category_hierarchy.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  color
  "${THIS_MODULE}"
  SOURCES color.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  covariance
  "${THIS_MODULE}"
  SOURCES covariance.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  database_query
  "${THIS_MODULE}"
  SOURCES database_query.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  descriptor
  "${THIS_MODULE}"
  SOURCES descriptor.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  descriptor_request
  "${THIS_MODULE}"
  SOURCES descriptor_request.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  descriptor_set
  "${THIS_MODULE}"
  SOURCES descriptor_set.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  detected_object
  "${THIS_MODULE}"
  SOURCES detected_object.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  detected_object_set
  "${THIS_MODULE}"
  SOURCES detected_object_set.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  detected_object_type
  "${THIS_MODULE}"
  SOURCES detected_object_type.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  essential_matrix
  "${THIS_MODULE}"
  SOURCES essential_matrix.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  feature
  "${THIS_MODULE}"
  SOURCES feature.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  feature_set
  "${THIS_MODULE}"
  SOURCES feature_set.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  feature_track_set
  "${THIS_MODULE}"
  SOURCES feature_track_set.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  fundamental_matrix
  "${THIS_MODULE}"
  SOURCES fundamental_matrix.cxx
  PRIVATE pybind11::pybind11
          vital
)

# Dropped with phase 5's import: the C++ types behind these did not come
# across, because nothing in VIAME reaches them and phase 5's prune lists
# them anyway. The python names go too; nothing imports them.

kwiver_add_python_library(
  geodesy
  "${THIS_MODULE}"
  SOURCES geodesy.cxx
  PRIVATE pybind11::pybind11
          vital
)



kwiver_add_python_library(
  geo_point
  "${THIS_MODULE}"
  SOURCES geo_point.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  geo_polygon
  "${THIS_MODULE}"
  SOURCES geo_polygon.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  homography
  "${THIS_MODULE}"
  SOURCES homography.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  homography_f2f
  "${THIS_MODULE}"
  SOURCES homography_f2f.cxx
  PRIVATE pybind11::pybind11
          vital
)


kwiver_add_python_library(
  iqr_feedback
  "${THIS_MODULE}"
  SOURCES iqr_feedback.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  landmark
  "${THIS_MODULE}"
  SOURCES landmark.cxx
  PRIVATE pybind11::pybind11
          vital)

kwiver_add_python_library(
  landmark_map
  "${THIS_MODULE}"
  SOURCES landmark_map.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  local_tangent_space
  "${THIS_MODULE}"
  SOURCES local_tangent_space.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  match_set
  "${THIS_MODULE}"
  SOURCES match_set.cxx
  PRIVATE pybind11::pybind11
          vital
)


kwiver_add_python_library(
  metadata
  "${THIS_MODULE}"
  SOURCES metadata.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  metadata_map
  "${THIS_MODULE}"
  SOURCES metadata_map.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  metadata_tags
  "${THIS_MODULE}"
  SOURCES metadata_tags.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  metadata_traits
  "${THIS_MODULE}"
  SOURCES metadata_traits.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  point
  "${THIS_MODULE}"
  SOURCES point.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  polygon
  "${THIS_MODULE}"
  SOURCES polygon.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  query_result
  "${THIS_MODULE}"
  SOURCES query_result.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  rotation
  "${THIS_MODULE}"
  SOURCES rotation.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  similarity
  "${THIS_MODULE}"
  SOURCES similarity.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  sfm_constraints
  "${THIS_MODULE}"
  SOURCES sfm_constraints.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  timestamp
  "${THIS_MODULE}"
  SOURCES timestamp.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  track
  "${THIS_MODULE}"
  SOURCES track.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  track_descriptor
  "${THIS_MODULE}"
  SOURCES track_descriptor.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  track_interval
  "${THIS_MODULE}"
  SOURCES track_interval.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  track_set
  "${THIS_MODULE}"
  SOURCES track_set.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  transform_2d
  "${THIS_MODULE}"
  SOURCES transform_2d.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  uid
  "${THIS_MODULE}"
  SOURCES uid.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  object_track_set
  "${THIS_MODULE}"
  SOURCES object_track_set.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  video_raw_image
  "${THIS_MODULE}"
  SOURCES video_raw_image.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  video_raw_metadata
  "${THIS_MODULE}"
  SOURCES video_raw_metadata.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  video_settings
  "${THIS_MODULE}"
  SOURCES video_settings.cxx
  PRIVATE pybind11::pybind11
          vital
)

#if(NOT SKBUILD)
#  kwiver_create_python_init(vital/types
#    types
#    activity_type
#    bounding_box
#    category_hierarchy
#    camera
#    camera_intrinsics
#    camera_map
#    camera_perspective
#    camera_perspective_map
#    camera_rpc
#    color
#    covariance
#    database_query
#    descriptor
#    descriptor_request
#    descriptor_set
#    # Next module is required by detected_object, so must be loaded first.
#    detected_object_type
#    detected_object
#    detected_object_set
#    essential_matrix
#    feature
#    feature_set
#    feature_track_set
#    fundamental_matrix
#    geodesy
#    geo_covariance
#    geo_MGRS
#    geo_polygon
#    geo_point
#    transform_2d
#    homography
#    homography_f2f
#    homography_f2w
#    iqr_feedback
#    landmark
#    landmark_map
#    rotation
#    match_set
#    mesh
#    metadata
#    metadata_map
#    metadata_tags
#    metadata_traits
#    point
#    polygon
#    query_result
#    similarity
#    sfm_constraints
#    timestamp
#    track
#    track_descriptor
#    track_interval
#    track_set
#    uid
#    object_track_set
#    # activity depends on timestamp, which must be loaded first
#    activity
#  )
#endif()
