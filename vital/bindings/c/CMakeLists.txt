# C interface library

include(CMakeDependentOption)

option(VITAL_ENABLE_C_LIB
  "Enable building of C interface library."
  OFF
  )
cmake_dependent_option( VITAL_C_LIB_FORCE_SHARED
  "Force C interface library to be build shared."
  OFF  # Default value
  "VITAL_ENABLE_C_LIB; NOT BUILD_SHARED_LIBS"
  OFF # State when dependency expression is false
  )


if( VITAL_ENABLE_C_LIB )

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

  if( VITAL_C_LIB_FORCE_SHARED )
    set(vital_force_shared_lib SHARED)
  endif()

  kwiver_install_headers(
    ${vital_c_public_headers}
    SUBDIR vital/c
    )

  kwiver_install_headers(
    ${CMAKE_CURRENT_BINARY_DIR}/vital_c_export.h
    SUBDIR vital/c
    NOPATH
    )

  kwiver_add_library( vital_c
    ${vital_force_shared_lib}
    ${vital_c_public_headers}
    ${vital_c_private_headers}
    ${vital_c_sources}
    )

  target_link_libraries( vital_c
    vital
    vital_apm
    )

endif( VITAL_ENABLE_C_LIB )
