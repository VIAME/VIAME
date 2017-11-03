#
# KPF parsing
#

if(fletch_ENABLED_YAMLCPP)
  find_package( yaml-cpp REQUIRED )
else()
  message(STATUS "To enable kpf support, please build or located yaml libraries in fletch")
  return()
endif()

set( sources
  kpf_packet.cxx
  kpf_parse_utils.cxx
  kpf_reader.cxx
  kpf_text_parser.cxx
  kpf_yaml_parser.cxx
  kpf_text_writer.cxx
  kpf_yaml_writer.cxx
)

set( public_headers
  kpf_packet.h
  kpf_canonical_types.h
  kpf_canonical_io.h
  kpf_reader.h
  kpf_parse_utils.h
  kpf_parser_base.h
  kpf_text_parser.h
  kpf_yaml_parser.h
  kpf_canonical_io_adapter_base.h
  kpf_canonical_io_adapter.h
  kpf_text_writer.h
  kpf_yaml_writer.h
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

target_include_directories(vital_kpf PRIVATE ${YAML_CPP_INCLUDE_DIR})

if(WIN32)
  target_compile_definitions(vital_kpf PRIVATE YAML_CPP_DLL)
endif()

target_link_libraries( vital_kpf
  PRIVATE        vital
                 vital_logger
                 vital_util
                 ${YAML_CPP_LIBRARIES}
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
