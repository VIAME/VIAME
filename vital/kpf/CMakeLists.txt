#
# KPF parsing
#

set( sources
  kpf_packet.cxx
  kpf_parse_utils.cxx
  kpf_parse.cxx
  kpf_text_parser.cxx
)

set( public_headers
  kpf_packet.h
  kpf_canonical_types.h
  kpf_canonical_io.h
  kpf_parse.h
  kpf_parse_utils.h
  kpf_parser_base.h
  kpf_text_parser.h
  kpf_canonical_io_adapter_base.h
  kpf_canonical_io_adapter.h
)

set( private_headers
  kpf_bounce_buffer.h
)

set( private_sources
  kpf_bounce_buffer.cxx
)

kwiver_add_library( vital_kpf
  ${sources}
  ${public_headers}
  ${private_sources}
)

target_link_libraries( vital_kpf
  PRIVATE        vital
                 vital_logger
                 vital_util
)

kwiver_install_headers(
  ${public_headers}
  SUBDIR vital/kpf
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_kpf_export.h
  NOPATH
  SUBDIR   vital/kpf
)

# if (KWIVER_ENABLE_TESTS)
#   add_subdirectory( tests )
# endif()

add_subdirectory( examples )
