set( track_oracle_aoi_utils_public_headers
  aoi_utils.h
)

set( track_oracle_aoi_utils_sources
  aoi_utils.cxx
)

kwiver_install_headers(
  ${track_oracle_aoi_utils_public_headers}
  SUBDIR             track_oracle/file_formats/aoi_utils
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_aoi_utils_export.h
  NOPATH SUBDIR             track_oracle/file_formats/aoi_utils
)

kwiver_add_library( track_oracle_aoi_utils
  ${track_oracle_aoi_utils_public_headers}
  ${track_oracle_aoi_utils_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_aoi_utils_export.h
)

target_link_libraries( track_oracle_aoi_utils
  PRIVATE              track_oracle
                       track_scorable_mgrs
                       vital_logger
                       vgl
)
