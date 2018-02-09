#
# KPF parsing
#


set( sources
  kpf_packet_header.cxx
  kpf_packet.cxx
  kpf_canonical_types.cxx
  kpf_parse_utils.cxx
  kpf_reader.cxx
  kpf_yaml_parser.cxx
  kpf_yaml_writer.cxx
  kpf_yaml_schemas.cxx
  kpf_exception.cxx
)

set( public_headers
  kpf_bounce_buffer.h
  kpf_packet_header.h
  kpf_packet.h
  kpf_canonical_types.h
  kpf_canonical_io.h
  kpf_reader.h
  kpf_parse_utils.h
  kpf_parser_base.h
  kpf_yaml_parser.h
  kpf_canonical_io_adapter_base.h
  kpf_canonical_io_adapter.h
  kpf_yaml_writer.h
  kpf_yaml_schemas.h
  kpf_exception.h
)

set( private_headers
  kpf_bounce_buffer.h
)

set( private_sources
  kpf_bounce_buffer.cxx
)

kwiver_add_library( kpf_yaml
  ${sources}
  ${public_headers}
  ${private_sources}
)

target_include_directories(kpf_yaml PRIVATE ${YAML_CPP_INCLUDE_DIR})

if(WIN32)
  target_compile_definitions(kpf_yaml PRIVATE YAML_CPP_DLL)
endif()

target_link_libraries( kpf_yaml
  PRIVATE        vital
                 vital_logger
                 vital_util
                 ${YAML_CPP_LIBRARIES}
  )

kwiver_install_headers(
  ${public_headers}
  SUBDIR arrows/kpf/yaml
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kpf_yaml_export.h
  NOPATH
  SUBDIR   arrows/kpf/yaml
)

if (KWIVER_ENABLE_TESTS)
   add_subdirectory( tests )
endif()
