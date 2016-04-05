# Build / install plugin containing VisCL algorithm implementations + support
# structures

find_package(Boost) ##+ temp

set( viscl_headers_public
  convert_image.h
  descriptor_set.h
  detect_features.h
  extract_descriptors.h
  feature_set.h
  image_container.h
  match_features.h
  match_set.h
  register_algorithms.h
  utils.h
  )
kwiver_install_headers(
  SUBDIR     arrows/algorithms/viscl
  ${viscl_headers_public}
  )
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/algorithms_viscl_export.h
  NOPATH   SUBDIR     arrows/algorithms/viscl
  )

set( viscl_sources
  convert_image.cxx
  descriptor_set.cxx
  detect_features.cxx
  extract_descriptors.cxx
  feature_set.cxx
  image_container.cxx
  match_features.cxx
  match_set.cxx
  register_algorithms.cxx
  utils.cxx
  )
kwiver_add_library( algorithms_viscl
  ${viscl_headers_public}
  ${viscl_sources}
  )
target_link_libraries( algorithms_viscl
  PUBLIC           algorithms_core
                   viscl_core viscl_tasks viscl_vxl
  )
algorithms_create_plugin( algorithms_viscl
  algorithm_plugin_interface.cxx
  )

if (ARROWS_ENABLE_TESTS)
  ## add_subdirectory(tests)
endif()
