find_package( yaml-cpp REQUIRED )

#
# KPF geometry
#

set( track_kpf_geom_public_headers
  track_kpf_geom.h
  file_format_kpf_geom.h
)

set( track_kpf_geom_sources
  file_format_kpf_geom.cxx
)

kwiver_install_headers(
  ${track_kpf_geom_public_headers}
  SUBDIR     track_oracle/file_formats/track_kpf_geom
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/track_kpf_geom_export.h
  NOPATH SUBDIR     track_oracle/file_formats/track_kpf_geom
)

kwiver_add_library( track_kpf_geom
  ${track_kpf_geom_public_headers}
  ${track_kpf_geom_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/track_kpf_geom_export.h
)

target_link_libraries( track_kpf_geom
  PUBLIC               track_oracle
                       data_terms
                       kwiver_algo_kpf
                       ${YAML_CPP_LIBRARIES}
  PRIVATE              vital_logger
                       vul
                       vgl
                       track_oracle_format_base
                       logging_map
                       kpf_utils
                       ${YAML_CPP_LIBRARIES}
)
