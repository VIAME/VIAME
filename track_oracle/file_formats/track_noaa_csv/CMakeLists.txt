#
# NOAA CSV tracks
#

set( track_noaa_csv_public_headers
  track_noaa_csv.h
  file_format_noaa_csv.h
)

set( track_noaa_csv_sources
  file_format_noaa_csv.cxx
)

kwiver_install_headers(
  ${track_noaa_csv_public_headers}
  SUBDIR     track_oracle/file_formats/track_noaa_csv
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_noaa_csv_export.h
  NOPATH SUBDIR     track_oracle/file_formats/track_noaa_csv
)

kwiver_add_library( track_noaa_csv
  ${track_noaa_csv_public_headers}
  ${track_noaa_csv_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_noaa_csv_export.h
)

target_link_libraries( track_noaa_csv
  PUBLIC               track_oracle
                       data_terms
                       track_oracle_format_base
  PRIVATE              vital_logger
                       track_oracle_tokenizers
                       vul
                       vgl
)
