include_directories("${CMAKE_CURRENT_BINARY_DIR}")

set( track_oracle_logging_map_public_headers
  logging_map.h
)

set( track_oracle_logging_map_sources
  logging_map.cxx
)

kwiver_install_headers(
  ${track_oracle_logging_map_public_headers}
  SUBDIR          track_oracle/utils
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/logging_map_export.h
  NOPATH SUBDIR          track_oracle/utils
)

kwiver_add_library( logging_map
  ${track_oracle_logging_map_public_headers}
  ${track_oracle_logging_map_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/logging_map_export.h
)

target_link_libraries( logging_map
  PUBLIC      vital_logger
)


set( track_oracle_tokenizers_public_headers
  tokenizers.h
)

set( track_oracle_tokenizers_sources
  tokenizers.cxx
)

kwiver_install_headers(
  ${track_oracle_tokenizers_public_headers}
  SUBDIR         track_oracle/utils
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_tokenizers_export.h
  SUBDIR         track_oracle/utils
  NOPATH
)

kwiver_add_library( track_oracle_tokenizers
  ${track_oracle_tokenizers_public_headers}
  ${track_oracle_tokenizers_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_tokenizers_export.h
)

target_link_libraries( track_oracle_tokenizers
  PRIVATE         vital_logger
)

