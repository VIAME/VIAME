# Build / Install plugin for serialization

set( headers_public

  )

set( sources

  )

kwiver_add_library( kwiver_algo_serialization
  ${headers_public}
  ${sources}
  )

target_link_libraries( kwiver_algo
  PUBLIC
  PRIVATE              kwiversys
  )

kwiver_install_headers(
  SUBDIR     arrows/serialization
  ${ocv_headers_public}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_serialization_export.h
  NOPATH   SUBDIR     arrows/serialization
  )

algorithms_create_plugin( kwiver_algo
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
