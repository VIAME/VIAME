# C interface library

set(vital_c_public_headers
  algorithm.h
  algorithm_plugin_manager.h
  camera.h
  camera_map.h
  common.h
  config_block.h
  error_handle.h
  image.h
  image_container.h
  image_container.hxx
  track.h
  track_set.h

  algo_def/convert_image.h
  algo_def/image_io.h
  algo_def/track_features.h
  )

set(vital_c_private_headers
  ${CMAKE_CURRENT_BINARY_DIR}/vital_c_export.h
  helpers/algorithm.h
  helpers/c_utils.h
  helpers/camera.h
  helpers/camera_map.h
  helpers/config_block.h
  helpers/image_container.h
  helpers/track.h
  helpers/track_set.h
  )

set(vital_c_sources
  algorithm.cxx
  algorithm_plugin_manager.cxx
  camera.cxx
  camera_map.cxx
  common.cxx
  config_block.cxx
  error_handle.cxx
  image.cxx
  image_container.cxx
  track.cxx
  track_set.cxx

  helpers/c_utils.cxx

  algo_def/convert_image.cxx
  algo_def/image_io.cxx
  algo_def/track_features.cxx
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
              vital_apm
  )
