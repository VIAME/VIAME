###
# algorithms/proj

set(proj_headers_public
  geo_map.h
  register_algorithms.h
  )

kwiver_install_headers(
  SUBDIR     arrows/proj
  ${proj_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_proj_export.h
  NOPATH   SUBDIR     arrows/proj
  )

set(proj_sources
  geo_map.cxx
  register_algorithms.cxx
  )

kwiver_add_library( kwiver_algo_proj
  ${proj_headers_public}
  ${proj_sources}
  )

target_link_libraries( kwiver_algo_proj
  PUBLIC               kwiver_algo
                       ${PROJ_LIBRARY}
  )

algorithms_create_plugin( kwiver_algo_proj
  algorithm_plugin_interface.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
