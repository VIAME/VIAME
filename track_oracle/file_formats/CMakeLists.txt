##
## format base
##

set( track_oracle_format_base_public_headers
  file_format_base.h
  file_format_type.h
  file_format_schema.h
  file_format_manager.h
)

set( track_oracle_format_base_sources
  file_format_base.cxx
  file_format_type.cxx
  file_format_schema.cxx
  templates/file_format_base_instances.cxx
)

kwiver_install_headers(
  ${track_oracle_format_base_public_headers}
  SUBDIR      track_oracle/file_formats
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_format_base_export.h
  NOPATH SUBDIR      track_oracle/file_formats
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
                      kwiversys
                      ${TinyXML_LIBRARY}
)
set_target_properties( track_oracle_format_base PROPERTIES CXX_VISIBILITY_PRESET default)

add_subdirectory( track_kw18 )

if (KWANT_ENABLE_SHAPELIB)
    add_subdirectory( track_apix )
    set(TRACK_ORACLE_APIX_LIBRARY track_apix)
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
add_subdirectory( track_noaa_csv )
if( KWIVER_ENABLE_KPF )
  add_subdirectory( track_kpf_geom )
  add_subdirectory( kpf_utils )
  add_subdirectory( track_filter_kpf_activity )
  set( TRACK_KPF_LIBRARIES track_kpf_geom kpf_utils track_filter_kpf_activity )
else()
  set( TRACK_KPF_LIBRARIES "" )
endif()

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
  SUBDIR        track_oracle/file_formats
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_oracle_file_formats_export.h
  SUBDIR        track_oracle/file_formats
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
                       ${TRACK_KPF_LIBRARIES}
                       track_noaa_csv
  PRIVATE              vital
                       vul
)
