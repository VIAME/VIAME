# Build / Install Multi-View Geometry arrow

set( plugin_mvg_headers
  algo/hierarchical_bundle_adjust.h
  algo/initialize_cameras_landmarks.h
  algo/initialize_cameras_landmarks_basic.h
  algo/triangulate_landmarks.h

  epipolar_geometry.h
  interpolate_camera.h
  necker_reverse.h
  metrics.h
  projected_track_set.h
  sfm_utils.h
  transform.h
  triangulate.h
  )

kwiver_install_headers(
  SUBDIR     arrows/mvg
  ${plugin_mvg_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_mvg_export.h
  NOPATH   SUBDIR     arrows/mvg
  )

set( plugin_mvg_sources
  algo/hierarchical_bundle_adjust.cxx
  algo/initialize_cameras_landmarks.cxx
  algo/initialize_cameras_landmarks_basic.cxx
  algo/triangulate_landmarks.cxx

  epipolar_geometry.cxx
  interpolate_camera.cxx
  necker_reverse.cxx
  metrics.cxx
  projected_track_set.cxx
  sfm_utils.cxx
  transform.cxx
  triangulate.cxx
  )

# Address MSVC fatal error C1128:
# number of sections exceeded object file format limit
if(MSVC)
  set_source_files_properties(
    triangulate.cxx
    PROPERTIES COMPILE_FLAGS "/bigobj"
    )
endif()


kwiver_add_library( kwiver_algo_mvg
  ${plugin_mvg_headers}
  ${plugin_mvg_sources}
  )

target_link_libraries( kwiver_algo_mvg
  PUBLIC               vital_algo
  PRIVATE              kwiver_algo_core
                       kwiversys
                       vital_config
  )

algorithms_create_plugin( kwiver_algo_mvg
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if (KWIVER_ENABLE_TOOLS)
  add_subdirectory(applets)
endif()
