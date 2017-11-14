project( kpf_example )

add_executable( kpf_example_simple kpf_example_simple.cxx )
target_link_libraries( kpf_example_simple vital_kpf )

add_executable( kpf_example kpf_example.cxx )
target_link_libraries( kpf_example vital_kpf )

add_executable( kpf_example_complex kpf_example_complex.cxx )
target_link_libraries( kpf_example_complex vital_kpf )

add_executable( kpf_text_reader kpf_text_reader.cxx )
target_link_libraries( kpf_text_reader vital_kpf )

add_executable( kpf_yaml_reader kpf_yaml_reader.cxx )
target_link_libraries( kpf_yaml_reader vital_kpf )

add_executable( yaml_parser yaml_parser.cxx )
target_link_libraries( yaml_parser ${YAML_CPP_LIBRARIES} )

add_executable( kpf_example_yaml kpf_example_yaml.cxx )
target_link_libraries( kpf_example_yaml vital_kpf ${YAML_CPP_LIBRARIES} )

if(WIN32)
  target_compile_definitions(kpf_example_simple PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_example PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_example_complex PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_text_reader PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_text_reader PRIVATE YAML_CPP_DLL)
  target_compile_definitions(yaml_parser PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_example_yaml PRIVATE YAML_CPP_DLL)
endif()
