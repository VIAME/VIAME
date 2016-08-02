OPTION(KWIVER_ENABLE_TRACK_ORACLE_EVENT_ADAPTER  "Enable track_oracle's event_adapter (disable until further notice)" OFF )
OPTION(KWIVER_ENABLE_TRACK_ORACLE_MGRS  "Enable track_oracle's lat/lon <-> MGRS capabilities (disable until further notice)" OFF )

if (KWIVER_ENABLE_TRACK_ORACLE_EVENT_ADAPTER)
  message( FATAL_ERROR "Building the event_adapter is blocked until VIBRANT lands" )
endif()
if (KWIVER_ENABLE_TRACK_MGRS)
  message( FATAL_ERROR "Building the MGRS code is blocked until lat/lon <-> MGRS capabilities are available" )
  set(TRACK_ORACLE_SCORABLE_MGRS_LIBRARY "scorable_mgrs")
  set(TRACK_ORACLE_SCORABLE_MGRS_DATA_TERM "scorable_mgrs_data_term")
else()
  set(TRACK_ORACLE_SCORABLE_MGRS_LIBRARY "" )
endif()

find_package( TinyXML REQUIRED )

add_subdirectory( aries_interface )
add_subdirectory( utils )

##
## Core track_oracle
##

set( track_oracle_public_headers
  track_oracle_api_types.h
  element_store.h
  element_store_base.h
  kwiver_io_base.h
  kwiver_io_base_data_io.h
  kwiver_io_helpers.h
  element_descriptor.h
  track_base.h
  track_oracle_core.h
  track_oracle_row_view.h
  track_oracle_frame_view.h
  track_field_host.h
  track_field_base.h
  track_field.h
  track_field_functor.h
  track_field_functor_library.h
  track_field_io_proxy.h
  schema_algorithm.h
  state_flags.h
)

set( track_oracle_private_headers
  kwiver_io_base_instantiation.h
  track_oracle_instantiation.h
  track_oracle_row_view_instantiation.h
  track_field_instantiation.h
  track_field_functor_instantiation.h
  track_field_output_specializations.h
  track_base_impl.h
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
  templates/track_oracle_instances_old_style.cxx
  )

if (KWIVER_ENABLE_TRACK_MGRS)
  set( track_oracle_sources
    ${track_oracle_sources}
    templates/track_oracle_instances_scorable_mgrs.cxx
    )
endif()

kwiver_install_headers(
  ${track_oracle_public_headers}
  SUBDIR       track_oracle
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_export.h
  SUBDIR       track_oracle
  NOPATH
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
                       ${Boost_THREAD_LIBRARY}
                       ${Boost_SYSTEM_LIBRARY}
  PRIVATE              vital_logger
                       vital
                       ${TRACK_ORACLE_SCORABLE_MGRS_LIBRARY}
                       track_oracle_tokenizers
                       vul
                       ${TinyXML_LIBRARY}
)
set_target_properties( track_oracle PROPERTIES CXX_VISIBILITY_PRESET default)
##
## format base
##

set( track_oracle_format_base_public_headers
  file_format_base.h
  file_format_type.h
  file_format_schema.h
)

set( track_oracle_format_base_sources
  file_format_base.cxx
  file_format_type.cxx
  file_format_schema.cxx
  templates/file_format_base_instances.cxx
)

kwiver_install_headers(
  ${track_oracle_format_base_public_headers}
  SUBDIR      track_oracle
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_format_base_export.h
  SUBDIR      track_oracle
  NOPATH
)

kwiver_add_library( track_oracle_format_base
  ${track_oracle_format_base_public_headers}
  ${track_oracle_format_base_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_format_base_export.h
)

target_link_libraries( track_oracle_format_base
  PUBLIC              track_oracle
                      vul
  PRIVATE             vital_logger
                      ${TinyXML_LIBRARY}
)
set_target_properties( track_oracle_format_base PROPERTIES CXX_VISIBILITY_PRESET default)

add_subdirectory( vibrant_descriptors )
add_subdirectory( data_terms )
add_subdirectory( example )
add_subdirectory( track_kw18 )

if (KWANT_ENABLE_SHAPELIB)
    add_subdirectory( track_apix )
    set(TRACK_ORACLE_APIX_LIBRARY track_apix)
endif()

##
## Can't build the event_adapter until we have vidtk event types
## via vibrant
##

if (KWIVER_ENABLE_TRACK_ORACLE_EVENT_ADAPTER)

set( track_oracle_event_adapter_public_headers
  event_adapter.h
)

set( track_oracle_event_adapter_sources
  event_adapter.cxx
)

kwiver_install_headers(
  ${track_oracle_event_adapter_public_headers}
  SUBDIR            track_oracle
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_event_adapter_export.h
  SUBDIR            track_oracle
  NOPATH
)

kwiver_add_library( track_oracle_event_adapter
  ${track_oracle_event_adapter_public_headers}
  ${track_oracle_event_adapter_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_event_adapter_export.h
)

target_link_libraries( track_oracle_event_adapter
  PUBLIC               logging_map
                       data_terms
  PRIVATE              track_oracle
                       scoring_aries_interface
                       vital_logger
                       vul
)

endif()

if (KWANT_ENABLE_4676)
  add_subdirectory( track_4676 )
  set( TRACK_ORACLE_4676_LIBRARY track_4676 )
endif()

add_subdirectory( track_comms_xml )
add_subdirectory( track_mitre_xml )
add_subdirectory( track_kwxml )
add_subdirectory( track_xgtf )
add_subdirectory( track_kwiver )
add_subdirectory( track_csv )
add_subdirectory( track_kst )
add_subdirectory( track_vatic )
add_subdirectory( track_vpd )
if (KWIVER_ENABLE_TRACK_ORACLE_MGRS)
  add_subdirectory( track_scorable_mgrs )
  add_subdirectory( aoi_utils )
endif()
add_subdirectory( track_e2at_callout )
if (KWIVER_ENABLE_TRACK_ORACLE_EVENT_ADAPTER)
  add_subdirectory( track_filter_kwe)
  set( TRACK_FILTER_KWE_LIBRARY track_filter_kwe )
else()
  set( TRACK_FILTER_KWE_LIBRARY "" )
endif()

set( track_oracle_file_formats_public_headers
  file_format_type.h
  file_format_manager.h
  schema_factory.h
)

set( track_oracle_file_formats_sources
  file_format_manager.cxx
  schema_factory.cxx
)

kwiver_install_headers(
  ${track_oracle_file_formats_public_headers}
  SUBDIR        track_oracle
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_file_formats_export.h
  SUBDIR        track_oracle
  NOPATH
)

kwiver_add_library( track_oracle_file_formats
  ${track_oracle_file_formats_public_headers}
  ${track_oracle_file_formats_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_file_formats_export.h
)

target_link_libraries( track_oracle_file_formats
  PUBLIC               track_oracle
                       track_kw18
                       track_mitre_xml
                       track_comms_xml
                       track_kwxml
                       track_xgtf
                       track_kst
                       track_vatic
                       track_vpd
                       track_e2at_callout
                       ${TRACK_4676_LIBRARY}
                       track_csv
                       track_kwiver
                       track_oracle_tokenizers
                       ${TRACK_FILTER_KWE_LIBRARY}
                       ${TRACK_ORACLE_APIX_LIBRARY}

  PRIVATE              ${Boost_THREAD_LIBRARY}
                       vital
                       vul
)
