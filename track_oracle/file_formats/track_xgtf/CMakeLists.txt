#
# xgtf tracks
#

set( track_xgtf_public_headers
  track_xgtf.h
  file_format_xgtf.h
)

set( track_xgtf_sources
  file_format_xgtf.cxx
)

kwiver_install_headers(
  ${track_xgtf_public_headers}
  SUBDIR        track_oracle/file_formats/track_xgtf
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_xgtf_export.h
  NOPATH SUBDIR        track_oracle/file_formats/track_xgtf
)

kwiver_add_library( track_xgtf
  ${track_xgtf_public_headers}
  ${track_xgtf_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_xgtf_export.h
)

target_link_libraries( track_xgtf
  PUBLIC               track_oracle
                       track_oracle_format_base
                       vgl
  PRIVATE              track_oracle_tokenizers
                       data_terms
                       logging_map
                       ${TinyXML_LIBRARY}
)
