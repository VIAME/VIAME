#
# kwiver tracks
#

set( track_kwiver_public_headers
  track_kwiver.h
  file_format_kwiver.h
)

set( track_kwiver_sources
  file_format_kwiver.cxx
)

kwiver_install_headers(
  ${track_kwiver_public_headers}
  SUBDIR        track_oracle/file_formats/track_kwiver
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_kwiver_export.h
  NOPATH SUBDIR        track_oracle/file_formats/track_kwiver
)

kwiver_add_library( track_kwiver
  ${track_kwiver_public_headers}
  ${track_kwiver_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_kwiver_export.h
)


target_link_libraries( track_kwiver
  PUBLIC               track_oracle
                       track_oracle_format_base
  PRIVATE              ${TinyXML_LIBRARY}
                       track_oracle_tokenizers
                       logging_map
)
