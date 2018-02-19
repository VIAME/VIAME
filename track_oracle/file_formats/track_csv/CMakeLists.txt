#
# Generic CSV tracks
#

set( track_csv_public_headers
  track_csv.h
  file_format_csv.h
)

set( track_csv_sources
  file_format_csv.cxx
)

kwiver_install_headers(
  ${track_csv_public_headers}
  SUBDIR        track_oracle/file_formats/track_csv
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_csv_export.h
  NOPATH SUBDIR        track_oracle/file_formats/track_csv
)

kwiver_add_library( track_csv
  ${track_csv_public_headers}
  ${track_csv_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_csv_export.h
)

target_link_libraries( track_csv
  PUBLIC               track_oracle_format_base
  PRIVATE              track_oracle
                       data_terms
                       vital_logger
                       ${TRACK_ORACLE_SCORABLE_MGRS_DATA_TERM}
                       track_oracle_tokenizers
                       vul
)
