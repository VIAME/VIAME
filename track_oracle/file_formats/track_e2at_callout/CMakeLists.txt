#
# e2at callouts
#

set( track_e2at_callout_public_headers
  track_e2at_callout.h
  file_format_e2at_callout.h
)

set( track_e2at_callout_sources
  file_format_e2at_callout.cxx
)

kwiver_install_headers(
  ${track_e2at_callout_public_headers}
  SUBDIR        track_oracle/file_formats/track_e2at_callout
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_e2at_callout_export.h
  NOPATH SUBDIR        track_oracle/file_formats/track_e2at_callout
)

kwiver_add_library( track_e2at_callout
  ${track_e2at_callout_public_headers}
  ${track_e2at_callout_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_e2at_callout_export.h
)

target_link_libraries( track_e2at_callout
  PUBLIC               track_oracle
                       track_oracle_format_base
  PRIVATE              data_terms
                       track_oracle_tokenizers
                       vital_logger
                       kwiversys
)
