# algorithms/proj

set(proj_headers_public
  geo_map.h
  register_algorithms.h
  )

kwiver_install_headers(
  SUBDIR     arrows/algorithms/proj
  ${proj_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/algorithms_proj_export.h
  NOPATH   SUBDIR     arrows/algorithms/proj
  )

set(proj_sources
  geo_map.cxx
  register_algorithms.cxx
  )
kwiver_add_library( algorithms_proj
  ${proj_headers_public}
  ${proj_sources}
  )
target_link_libraries( algorithms_proj
  PUBLIC               algorithms_core
                       ${PROJ4_LIBRARY}
  )
algorithms_create_plugin( algorithms_proj
  algorithm_plugin_interface.cxx
  )

if (ARROWS_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
