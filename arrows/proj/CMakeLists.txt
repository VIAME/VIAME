###
# algorithms/proj

set(CMAKE_FOLDER "Arrows/PROJ")

set(proj_headers_public
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
  geo_conv.cxx
  )

kwiver_add_library( kwiver_algo_proj
  ${proj_headers_public}
  ${proj_sources}
  )

target_include_directories( kwiver_algo_proj
  SYSTEM PRIVATE ${PROJ_INCLUDE_DIR}
  )

target_link_libraries( kwiver_algo_proj
  PUBLIC               vital_algo
                       ${PROJ_LIBRARY}
  )

algorithms_create_plugin( kwiver_algo_proj
  register_algorithms.cxx
  )
