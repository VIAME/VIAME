#
# kst tracks
#

set( track_kst_public_headers
  track_kst.h
  file_format_kst.h
)

set( track_kst_sources
  file_format_kst.cxx
)

kwiver_install_headers(
  ${track_kst_public_headers}
  SUBDIR        track_oracle/file_formats/track_kst
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_kst_export.h
  NOPATH SUBDIR        track_oracle/file_formats/track_kst
)

kwiver_add_library( track_kst
  ${track_kst_public_headers}
  ${track_kst_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_kst_export.h
)

target_link_libraries( track_kst
  PUBLIC               track_oracle
                       track_oracle_format_base
                       vgl
                       data_terms
  PRIVATE              vital_logger
                       vul
)
