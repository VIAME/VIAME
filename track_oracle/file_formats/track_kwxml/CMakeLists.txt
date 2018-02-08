#
# kwxml tracks
#

set( track_kwxml_public_headers
  track_kwxml.h
  file_format_kwxml.h
)

set( track_kwxml_sources
  file_format_kwxml.cxx
)

kwiver_install_headers(
  ${track_kwxml_public_headers}
  SUBDIR        track_oracle/file_formats/track_kwxml
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_kwxml_export.h
  NOPATH SUBDIR        track_oracle/file_formats/track_kwxml
)

kwiver_add_library( track_kwxml
  ${track_kwxml_public_headers}
  ${track_kwxml_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_kwxml_export.h
)

target_link_libraries( track_kwxml
  PUBLIC               track_oracle
                       track_oracle_format_base
                       vgl
                       vibrant_descriptors
                       data_terms
  PRIVATE              logging_map
                       scoring_aries_interface
                       vital_logger
                       track_oracle_tokenizers
                       ${TinyXML_LIBRARY}
)
