# Build / Install Multi-View Geometry arrow

set( plugin_mvg_headers
  algo/initialize_cameras_landmarks.h

  sfm_utils.h
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
  algo/initialize_cameras_landmarks.cxx

  sfm_utils.cxx
  )


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
#  add_subdirectory(tests)
endif()

if (KWIVER_ENABLE_TOOLS)
  add_subdirectory(applets)
endif()
