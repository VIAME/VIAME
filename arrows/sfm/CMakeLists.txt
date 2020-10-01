# Build / Install Structure-from-Motion arrow

set( plugin_sfm_headers
  algo/initialize_cameras_landmarks.h

  sfm_utils.h
  )

kwiver_install_headers(
  SUBDIR     arrows/sfm
  ${plugin_sfm_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_sfm_export.h
  NOPATH   SUBDIR     arrows/sfm
  )

set( plugin_sfm_sources
  algo/initialize_cameras_landmarks.cxx

  sfm_utils.cxx
  )


kwiver_add_library( kwiver_algo_sfm
  ${plugin_sfm_headers}
  ${plugin_sfm_sources}
  )

target_link_libraries( kwiver_algo_sfm
  PUBLIC               vital_algo
  PRIVATE              kwiver_algo_core
                       kwiversys
                       vital_config
  )

algorithms_create_plugin( kwiver_algo_sfm
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
#  add_subdirectory(tests)
endif()

if (KWIVER_ENABLE_TOOLS)
  add_subdirectory(applets)
endif()
