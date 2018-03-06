if(fletch_ENABLED_YAMLCPP)
  find_package( yaml-cpp REQUIRED )
else()
  message(STATUS "Cannot build track_oracle KPF support without YAML in Fletch")
  return()
endif()

#
# KPF track_oracle utilities
#

set( kpf_utils_public_headers
  kpf_utils.h
  kpf_cset_io.h
)

set( kpf_utils_sources
  kpf_utils.cxx
  kpf_cset_io.cxx
  kpf_cset_instances.cxx
)

kwiver_install_headers(
  ${kpf_utils_public_headers}
  SUBDIR track_oracle/file_formats/kpf_utils
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kpf_utils_export.h
  NOPATH SUBDIR   track_oracle/file_formats/kpf_utils
)

kwiver_add_library( kpf_utils
  ${kpf_utils_public_headers}
  ${kpf_utils_sources}
  ${CMAKE_CURRENT_BINARY_DIR}/kpf_utils_export.h
)

target_link_libraries( kpf_utils
  PUBLIC               track_oracle
                       data_terms
                       kwiver_algo_kpf
  PRIVATE              vital_logger
                       logging_map
                       kwiversys
                       ${YAML_CPP_LIBRARIES}
)
