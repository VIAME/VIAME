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
  utils.h
  )
kwiver_install_headers(
  ${viscl_headers_public}
  SUBDIR     arrows/viscl
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_viscl_export.h
  NOPATH   SUBDIR     arrows/viscl
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
  utils.cxx
  )

kwiver_add_library( kwiver_algo_viscl
  ${viscl_headers_public}
  ${viscl_sources}
  )

target_link_libraries( kwiver_algo_viscl
  PUBLIC           vital_algo
                   viscl_core viscl_tasks viscl_vxl
  )

algorithms_create_plugin( kwiver_algo_viscl
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  ## add_subdirectory(tests)
endif()
