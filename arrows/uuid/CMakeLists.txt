# Build / Install Plugin containing UUID algorithm implementations

set( plugin_headers
  uuid_factory_uuid.h
  )

kwiver_install_headers(
  SUBDIR     arrows/uuid
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_uuid_export.h
  NOPATH   SUBDIR     arrows/uuid
  )

set( plugin_sources
  uuid_factory_uuid.cxx
  )

kwiver_add_library( kwiver_algo_uuid
  ${plugin_headers}
  ${plugin_sources}
  )

target_link_libraries( kwiver_algo_uuid
  PUBLIC               vital_algo
  PRIVATE              vital_config
                       ${KWIVER_UUID_LIBRARY}
  )

algorithms_create_plugin( kwiver_algo_uuid
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
