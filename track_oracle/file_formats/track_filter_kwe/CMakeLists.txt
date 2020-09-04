#
# kwe track filter
#

set( track_filter_kwe_public_headers
  track_filter_kwe.h
)

set( track_filter_kwe_sources
  track_filter_kwe.cxx
)

kwiver_install_headers(
  ${track_filter_kwe_public_headers}
  SUBDIR        track_oracle/file_formats/track_filter_kwe
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_filter_kwe_export.h
  NOPATH SUBDIR        track_oracle/file_formats/track_filter_kwe
)

kwiver_add_library( track_filter_kwe
  ${track_filter_kwe_public_headers}
  ${track_filter_kwe_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_filter_kwe_export.h
)

target_link_libraries( track_filter_kwe
  PUBLIC               track_oracle_event_adapter
  PRIVATE              track_oracle
                       data_terms
                       scoring_aries_interface
                       vital_logger
                       logging_map
                       vul
)
