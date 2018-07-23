set( track_oracle_public_headers
  track_oracle_api_types.h
  element_store.h
  element_store_base.h
  kwiver_io_base.h
  kwiver_io_base_data_io.h
  kwiver_io_helpers.h
  element_descriptor.h
  track_base.h
  track_base_impl.h
  track_oracle_core.h
  track_oracle_row_view.h
  track_oracle_frame_view.h
  track_field_host.h
  track_field_base.h
  track_field.h
  track_field_io_proxy.h
  schema_algorithm.h
  state_flags.h
)

set( track_oracle_private_headers
  kwiver_io_base_instantiation.h
  track_oracle_instantiation.h
  track_oracle_row_view_instantiation.h
  track_field_instantiation.h
  track_field_output_specializations.h
  track_oracle_core_impl.h
)

set( track_oracle_sources
  track_oracle_api_types.cxx
  element_store.txx
  element_store_base.cxx
  kwiver_io_base.txx
  kwiver_io_base_data_io.cxx
  kwiver_io_helpers.cxx
  element_descriptor.cxx
  track_base_impl.cxx
  track_oracle_core.cxx
  track_oracle_core.txx
  track_oracle_row_view.cxx
  track_oracle_row_view.txx
  track_oracle_frame_view.cxx
  track_field_host.cxx
  track_field_base.cxx
  track_field.txx
  track_oracle_core_impl.cxx
  track_oracle_core_impl.txx
  track_field_io_proxy.txx
  track_field_output_specializations.cxx
  schema_algorithm.cxx
  state_flags.cxx
  track_oracle_instances_old_style.cxx
  )

if (KWIVER_ENABLE_TRACK_MGRS)
  set( track_oracle_sources
    ${track_oracle_sources}
    track_oracle_instances_scorable_mgrs.cxx
    )
endif()

kwiver_install_headers(
  ${track_oracle_public_headers}
  SUBDIR       track_oracle/core
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_export.h
  NOPATH SUBDIR       track_oracle/core
)

kwiver_add_library( track_oracle
  ${track_oracle_public_headers}
  ${track_oracle_private_headers}
  ${track_oracle_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_export.h
)


target_link_libraries( track_oracle
  PUBLIC               scoring_aries_interface
                       vnl
                       vgl
  PRIVATE              vital_logger
                       vital
                       ${TRACK_ORACLE_SCORABLE_MGRS_LIBRARY}
                       track_oracle_tokenizers
                       vul
                       ${TinyXML_LIBRARY}
)
set_target_properties( track_oracle PROPERTIES CXX_VISIBILITY_PRESET default)
