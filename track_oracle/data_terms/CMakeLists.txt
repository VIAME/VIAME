#
# track oracle data terms
#

set( data_terms_public_headers
  data_terms_common.h
  data_term_tmp_utils.h
  data_terms.h
)

set( data_terms_sources
  data_terms.cxx
  data_term_instances.cxx
)

kwiver_install_headers(
  ${data_terms_public_headers}
  SUBDIR        track_oracle/data_terms
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/data_terms_export.h
  NOPATH SUBDIR        track_oracle/data_terms
)

kwiver_add_library( data_terms
  ${data_terms_public_headers}
  ${data_terms_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/data_terms_export.h
)

target_link_libraries( data_terms
  PUBLIC               track_oracle
                       ${TinyXML_LIBRARY}
  PRIVATE              scoring_aries_interface
                       vital_logger
                       vital
  INTERFACE            vgl
)
set_target_properties( data_terms PROPERTIES CXX_VISIBILITY_PRESET default)
