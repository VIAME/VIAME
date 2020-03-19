project(vital_python_types)

include_directories(${pybind11_INCLUDE_DIR})

set( vital_python_headers
     image.h
     image_container.h
  )

set( vital_python_sources
     image.cxx
     image_container.cxx
     types_module.cxx
   )

kwiver_add_python_library(types
  vital/types
  ${vital_python_headers}
  ${vital_python_sources}
  )

target_link_libraries(python-vital.types-types
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(activity
  vital/types
  activity.cxx
  )
target_link_libraries(python-vital.types-activity
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(activity_type
  vital/types
  activity_type.cxx
  )
target_link_libraries(python-vital.types-activity_type
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(bounding_box
  vital/types
  bounding_box.cxx)
target_link_libraries(python-vital.types-bounding_box
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(camera
  vital/types
  camera.cxx)
target_link_libraries(python-vital.types-camera
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(camera_intrinsics
  vital/types
  camera_intrinsics.cxx)
target_link_libraries(python-vital.types-camera_intrinsics
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(camera_map
  vital/types
  camera_map.cxx)
target_link_libraries(python-vital.types-camera_map
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(camera_perspective
  vital/types
  camera_perspective.cxx)
target_link_libraries(python-vital.types-camera_perspective
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(camera_rpc
  vital/types
  camera_rpc.cxx)
target_link_libraries(python-vital.types-camera_rpc
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(category_hierarchy
  vital/types
  category_hierarchy.cxx)
target_link_libraries(python-vital.types-category_hierarchy
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(color
  vital/types
  color.cxx)
target_link_libraries(python-vital.types-color
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(covariance
  vital/types
  covariance.cxx)
target_link_libraries(python-vital.types-covariance
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(database_query
  vital/types
  database_query.cxx)
target_link_libraries(python-vital.types-database_query
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(descriptor
  vital/types
  descriptor.cxx)
target_link_libraries(python-vital.types-descriptor
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(descriptor_request
  vital/types
  descriptor_request.cxx)
target_link_libraries(python-vital.types-descriptor_request
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(descriptor_set
  vital/types
  descriptor_set.cxx)
target_link_libraries(python-vital.types-descriptor_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(detected_object
  vital/types
  detected_object.cxx)
target_link_libraries(python-vital.types-detected_object
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(detected_object_set
  vital/types
  detected_object_set.cxx)
target_link_libraries(python-vital.types-detected_object_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(detected_object_type
  vital/types
  detected_object_type.cxx)
target_link_libraries(python-vital.types-detected_object_type
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(feature
  vital/types
  feature.cxx)
target_link_libraries(python-vital.types-feature
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(feature_set
  vital/types
  feature_set.cxx)
target_link_libraries(python-vital.types-feature_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(fundamental_matrix
  vital/types
  fundamental_matrix.cxx)
target_link_libraries(python-vital.types-fundamental_matrix
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(geodesy
  vital/types
  geodesy.cxx)
target_link_libraries(python-vital.types-geodesy
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(geo_MGRS
  vital/types
  geo_MGRS.cxx)
target_link_libraries(python-vital.types-geo_MGRS
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(geo_covariance
  vital/types
  geo_covariance.cxx)
target_link_libraries(python-vital.types-geo_covariance
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(geo_point
  vital/types
  geo_point.cxx)
target_link_libraries(python-vital.types-geo_point
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(geo_polygon
  vital/types
  geo_polygon.cxx)
target_link_libraries(python-vital.types-geo_polygon
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(homography
  vital/types
  homography.cxx)
target_link_libraries(python-vital.types-homography
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(homography_f2f
  vital/types
  homography_f2f.cxx)
target_link_libraries(python-vital.types-homography_f2f
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(homography_f2w
  vital/types
  homography_f2w.cxx)
target_link_libraries(python-vital.types-homography_f2w
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(iqr_feedback
  vital/types
  iqr_feedback.cxx)
target_link_libraries(python-vital.types-iqr_feedback
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(landmark
  vital/types
  landmark.cxx)
target_link_libraries(python-vital.types-landmark
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(local_cartesian
  vital/types
  local_cartesian.cxx)
target_link_libraries(python-vital.types-local_cartesian
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(local_geo_cs
  vital/types
  local_geo_cs.cxx)
target_link_libraries(python-vital.types-local_geo_cs
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(metadata
  vital/types
  metadata.cxx)
target_link_libraries(python-vital.types-metadata
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(metadata_map
  vital/types
  metadata_map.cxx)
target_link_libraries(python-vital.types-metadata_map
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(metadata_tags
  vital/types
  metadata_tags.cxx)
target_link_libraries(python-vital.types-metadata_tags
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(metadata_traits
  vital/types
  metadata_traits.cxx)
target_link_libraries(python-vital.types-metadata_traits
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(point
  vital/types
  point.cxx)
target_link_libraries(python-vital.types-point
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(polygon
  vital/types
  polygon.cxx)
target_link_libraries(python-vital.types-polygon
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(query_result
  vital/types
  query_result.cxx)
target_link_libraries(python-vital.types-query_result
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(rotation
  vital/types
  rotation.cxx)
target_link_libraries(python-vital.types-rotation
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(similarity
  vital/types
  similarity.cxx)
target_link_libraries(python-vital.types-similarity
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(sfm_constraints
  vital/types
  sfm_constraints.cxx)
target_link_libraries(python-vital.types-sfm_constraints
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(timestamp
  vital/types
  timestamp.cxx)
target_link_libraries(python-vital.types-timestamp
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(track
  vital/types
  track.cxx)
target_link_libraries(python-vital.types-track
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(track_descriptor
  vital/types
  track_descriptor.cxx)
target_link_libraries(python-vital.types-track_descriptor
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(track_interval
  vital/types
  track_interval.cxx)
target_link_libraries(python-vital.types-track_interval
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(track_set
  vital/types
  track_set.cxx)
target_link_libraries(python-vital.types-track_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(transform_2d
  vital/types
  transform_2d.cxx)
target_link_libraries(python-vital.types-transform_2d
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(uid
  vital/types
  uid.cxx)
target_link_libraries(python-vital.types-uid
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(object_track_set
  vital/types
  object_track_set.cxx)
target_link_libraries(python-vital.types-object_track_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

if(NOT SKBUILD)
  kwiver_create_python_init(vital/types
    types
    activity_type
    bounding_box
    category_hierarchy
    camera
    camera_intrinsics
    camera_map
    camera_perspective
    camera_rpc
    color
    covariance
    database_query
    descriptor
    descriptor_request
    descriptor_set
    # Next module is required by detected_object, so must be loaded first.
    detected_object_type
    detected_object
    detected_object_set
    feature
    feature_set
    fundamental_matrix
    geodesy
    geo_covariance
    geo_MGRS
    geo_polygon
    geo_point
    transform_2d
    homography
    homography_f2f
    homography_f2w
    iqr_feedback
    landmark
    local_cartesian
    rotation
    local_geo_cs
    metadata
    metadata_map
    metadata_tags
    metadata_traits
    point
    polygon
    query_result
    similarity
    sfm_constraints
    timestamp
    track
    track_descriptor
    track_interval
    track_set
    uid
    object_track_set
    activity
  )
endif()
