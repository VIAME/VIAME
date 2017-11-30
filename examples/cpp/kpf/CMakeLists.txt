project( kpf_example )

find_package( yaml-cpp REQUIRED )

add_executable( kpf_example_simple kpf_example_simple.cxx )
target_link_libraries( kpf_example_simple
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

add_executable( kpf_example_complex kpf_example_complex.cxx )
target_link_libraries( kpf_example_complex
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

add_executable( kpf_yaml_reader kpf_yaml_reader.cxx )
target_link_libraries( kpf_yaml_reader
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

add_executable( yaml_parser yaml_parser.cxx )
target_link_libraries( yaml_parser
  ${YAML_CPP_LIBRARIES} )

add_executable( kpf_example_yaml kpf_example_yaml.cxx )
target_link_libraries( kpf_example_yaml
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

if(WIN32)
  target_compile_definitions(kpf_example_simple PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_example_complex PRIVATE YAML_CPP_DLL)
  target_compile_definitions(yaml_parser PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_example_yaml PRIVATE YAML_CPP_DLL)
endif()
