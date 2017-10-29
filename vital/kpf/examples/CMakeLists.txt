project( kpf_example )

add_executable( kpf_example_simple kpf_example_simple.cxx )
target_link_libraries( kpf_example_simple vital_kpf )

add_executable( kpf_example kpf_example.cxx )
target_link_libraries( kpf_example vital_kpf )

add_executable( kpf_example_complex kpf_example_complex.cxx )
target_link_libraries( kpf_example_complex vital_kpf )

add_executable( kpf_text_reader kpf_text_reader.cxx )
target_link_libraries( kpf_text_reader vital_kpf )

add_executable( yaml_parser yaml_parser.cxx )
target_link_libraries( yaml_parser ${YAML_CPP_LIBRARIES} )
