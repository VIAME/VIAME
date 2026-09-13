###
# `kwiver.vital.types`
#
# The bindings live beside the C++ they bind since P8-T01, rather than in
# `python/kwiver/vital/types`, which is what the copy from kwiver left. Each
# file keeps its name with a `_python` suffix -- the convention
# `library/file_io/opencv_yaml_python.cxx` already set -- because 47 of them
# would otherwise collide with the C++ source of the very type they bind.
#
# **The module path is unchanged.** `viame_add_python_library` takes it as an
# argument rather than deriving it from the source location, so
# `kwiver.vital.types.bounding_box` is still `kwiver.vital.types.bounding_box`
# and the four files in the tree that import a submodule by name keep working.
# `tests/library/core_types/test_python_types.py` holds that surface to what
# it was: 106 classes and 965 members, recorded before this moved.
##

# Two settings that `python/CMakeLists.txt` made and this directory does not
# inherit, because both are directory scoped and the move changed the
# directory. Neither failure is obvious from its symptom.
#
#   `kwiver_python_package` decides the top package these install into. It
#   defaults to the project name, so without this every module landed in
#   `site-packages/viame/vital/types` and nothing could import
#   `kwiver.vital.types` at all.
#
#   `${PYTHON_LIBRARIES}` on every module, because VIAME links with
#   `-Wl,--no-undefined` and an extension module leaves the interpreter's
#   symbols to be resolved at import. In `python/` the flag was stripped from
#   the directory's link flags instead; linking libpython is what
#   `library/file_io` already does for `_opencv_yaml`, and it does not weaken
#   the check for the C++ in this directory the way stripping the flag would.
set( kwiver_python_package "kwiver" )

set( THIS_MODULE vital/types )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/types_init.py "${THIS_MODULE}" __init__ )

set( vital_python_headers
     image_python.h
     image_container_python.h
  )

set( vital_python_sources
     image_python.cxx
     image_container_python.cxx
     types_module_python.cxx
   )

viame_add_python_library(
  types
  "${THIS_MODULE}"
  SOURCES ${vital_python_headers}
          ${vital_python_sources}
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  activity
  "${THIS_MODULE}"
  SOURCES activity_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  activity_type
  "${THIS_MODULE}"
  SOURCES activity_type_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  bounding_box
  "${THIS_MODULE}"
  SOURCES bounding_box_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  camera
  "${THIS_MODULE}"
  SOURCES camera_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  camera_intrinsics
  "${THIS_MODULE}"
  SOURCES camera_intrinsics_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  camera_map
  "${THIS_MODULE}"
  SOURCES camera_map_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  camera_perspective
  "${THIS_MODULE}"
  SOURCES camera_perspective_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  camera_perspective_map
  "${THIS_MODULE}"
  SOURCES camera_perspective_map_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
           vital
  )

viame_add_python_library(
  camera_rpc
  "${THIS_MODULE}"
  SOURCES camera_rpc_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  category_hierarchy
  "${THIS_MODULE}"
  SOURCES category_hierarchy_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  color
  "${THIS_MODULE}"
  SOURCES color_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  covariance
  "${THIS_MODULE}"
  SOURCES covariance_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  database_query
  "${THIS_MODULE}"
  SOURCES database_query_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  descriptor
  "${THIS_MODULE}"
  SOURCES descriptor_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  descriptor_request
  "${THIS_MODULE}"
  SOURCES descriptor_request_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  descriptor_set
  "${THIS_MODULE}"
  SOURCES descriptor_set_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  detected_object
  "${THIS_MODULE}"
  SOURCES detected_object_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  detected_object_set
  "${THIS_MODULE}"
  SOURCES detected_object_set_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  detected_object_type
  "${THIS_MODULE}"
  SOURCES detected_object_type_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  essential_matrix
  "${THIS_MODULE}"
  SOURCES essential_matrix_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  feature
  "${THIS_MODULE}"
  SOURCES feature_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  feature_set
  "${THIS_MODULE}"
  SOURCES feature_set_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  feature_track_set
  "${THIS_MODULE}"
  SOURCES feature_track_set_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  fundamental_matrix
  "${THIS_MODULE}"
  SOURCES fundamental_matrix_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

# Dropped with phase 5's import: the C++ types behind these did not come
# across, because nothing in VIAME reaches them and phase 5's prune lists
# them anyway. The python names go too; nothing imports them.

viame_add_python_library(
  geodesy
  "${THIS_MODULE}"
  SOURCES geodesy_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)



viame_add_python_library(
  geo_point
  "${THIS_MODULE}"
  SOURCES geo_point_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  geo_polygon
  "${THIS_MODULE}"
  SOURCES geo_polygon_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  homography
  "${THIS_MODULE}"
  SOURCES homography_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  homography_f2f
  "${THIS_MODULE}"
  SOURCES homography_f2f_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)


viame_add_python_library(
  iqr_feedback
  "${THIS_MODULE}"
  SOURCES iqr_feedback_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  landmark
  "${THIS_MODULE}"
  SOURCES landmark_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital)

viame_add_python_library(
  landmark_map
  "${THIS_MODULE}"
  SOURCES landmark_map_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  local_tangent_space
  "${THIS_MODULE}"
  SOURCES local_tangent_space_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  match_set
  "${THIS_MODULE}"
  SOURCES match_set_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)


viame_add_python_library(
  metadata
  "${THIS_MODULE}"
  SOURCES metadata_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  metadata_map
  "${THIS_MODULE}"
  SOURCES metadata_map_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  metadata_tags
  "${THIS_MODULE}"
  SOURCES metadata_tags_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  metadata_traits
  "${THIS_MODULE}"
  SOURCES metadata_traits_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  point
  "${THIS_MODULE}"
  SOURCES point_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  polygon
  "${THIS_MODULE}"
  SOURCES polygon_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  query_result
  "${THIS_MODULE}"
  SOURCES query_result_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  rotation
  "${THIS_MODULE}"
  SOURCES rotation_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  similarity
  "${THIS_MODULE}"
  SOURCES similarity_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  sfm_constraints
  "${THIS_MODULE}"
  SOURCES sfm_constraints_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  timestamp
  "${THIS_MODULE}"
  SOURCES timestamp_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  track
  "${THIS_MODULE}"
  SOURCES track_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  track_descriptor
  "${THIS_MODULE}"
  SOURCES track_descriptor_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  track_interval
  "${THIS_MODULE}"
  SOURCES track_interval_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  track_set
  "${THIS_MODULE}"
  SOURCES track_set_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  transform_2d
  "${THIS_MODULE}"
  SOURCES transform_2d_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  uid
  "${THIS_MODULE}"
  SOURCES uid_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  object_track_set
  "${THIS_MODULE}"
  SOURCES object_track_set_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  video_raw_image
  "${THIS_MODULE}"
  SOURCES video_raw_image_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  video_raw_metadata
  "${THIS_MODULE}"
  SOURCES video_raw_metadata_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

viame_add_python_library(
  video_settings
  "${THIS_MODULE}"
  SOURCES video_settings_python.cxx
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
)

#if(NOT SKBUILD)
#  viame_create_python_init(vital/types
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
