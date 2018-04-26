#
# mitre XML tracks
#

set( track_mitre_xml_public_headers
  track_mitre_xml.h
  file_format_mitre_xml.h
)

set( track_mitre_xml_sources
  file_format_mitre_xml.cxx
)

kwiver_install_headers(
  ${track_mitre_xml_public_headers}
  SUBDIR        track_oracle/file_formats/track_mitre_xml
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_mitre_xml_export.h
  NOPATH SUBDIR        track_oracle/file_formats/track_mitre_xml
)

kwiver_add_library( track_mitre_xml
  ${track_mitre_xml_public_headers}
  ${track_mitre_xml_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_mitre_xml_export.h
)

target_link_libraries( track_mitre_xml
  PUBLIC               track_oracle
                       track_oracle_format_base
                       vgl
  PRIVATE              track_oracle_tokenizers
                       vital_logger
                       ${TinyXML_LIBRARY}
)
