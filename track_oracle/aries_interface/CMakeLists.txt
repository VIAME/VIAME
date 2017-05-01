set( aries_interface_public_headers
  aries_interface.h
)

set( aries_interface_sources
  aries_interface.cxx
)

kwiver_install_headers(
  ${aries_interface_public_headers}
  SUBDIR     track_oracle/aries_interface
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/scoring_aries_interface_export.h
  NOPATH SUBDIR     track_oracle/aries_interface
)

kwiver_add_library( scoring_aries_interface
  ${aries_interface_public_headers}
  ${aries_interface_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/scoring_aries_interface_export.h
)

target_link_libraries( scoring_aries_interface
  PRIVATE           vital_logger
)
