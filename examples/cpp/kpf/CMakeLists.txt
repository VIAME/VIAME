project( kpf_example )

find_package( yaml-cpp REQUIRED )

add_executable( kpf_example_simple kpf_example_simple.cxx )
target_include_directories( kpf_example_simple PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries( kpf_example_simple
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

add_executable( kpf_example_simple_classifier kpf_example_simple_classifier.cxx )
target_include_directories( kpf_example_simple_classifier PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries( kpf_example_simple_classifier
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)


add_executable( kpf_example_complex kpf_example_complex.cxx )
target_include_directories( kpf_example_complex PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries( kpf_example_complex
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

add_executable( kpf_example_activity kpf_example_activity.cxx )
target_include_directories( kpf_example_activity PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries( kpf_example_activity
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

add_executable( kpf_generic_reader kpf_generic_reader.cxx )
target_include_directories( kpf_generic_reader PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries( kpf_generic_reader
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

add_executable( kpf_yaml_reader kpf_yaml_reader.cxx )
target_include_directories( kpf_yaml_reader PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries( kpf_yaml_reader
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES}
)

add_executable( yaml_parser yaml_parser.cxx )
target_include_directories( yaml_parser PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries( yaml_parser
  ${YAML_CPP_LIBRARIES} )

add_executable( kpf_yaml_copy kpf_yaml_copy.cxx )
target_include_directories( kpf_yaml_copy PRIVATE ${YAML_CPP_INCLUDE_DIR})
target_link_libraries( kpf_yaml_copy
  kwiver_algo_kpf
  kpf_yaml
  ${YAML_CPP_LIBRARIES} )

if(WIN32)
  target_compile_definitions(kpf_example_simple PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_example_complex PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_example_activity PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_yaml_reader PRIVATE YAML_CPP_DLL)
  target_compile_definitions(kpf_yaml_copy PRIVATE YAML_CPP_DLL)
  target_compile_definitions(yaml_parser PRIVATE YAML_CPP_DLL)
endif()
