#
# comms XML tracks
#

set( track_comms_xml_public_headers
  track_comms_xml.h
  file_format_comms_xml.h
)

set( track_comms_xml_sources
  file_format_comms_xml.cxx
)

kwiver_install_headers(
  ${track_comms_xml_public_headers}
  SUBDIR         track_oracle/file_formats/track_comms_xml
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_comms_xml_export.h
  NOPATH SUBDIR     track_oracle/file_formats/track_comms_xml
)

kwiver_add_library( track_comms_xml
  ${track_comms_xml_public_headers}
  ${track_comms_xml_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_comms_xml_export.h
)

target_link_libraries( track_comms_xml
  PUBLIC               track_oracle
                       data_terms
                       track_oracle_format_base
                       vgl
  PRIVATE              vital_logger
                       ${TinyXML_LIBRARY}
                       track_oracle_tokenizers
)
