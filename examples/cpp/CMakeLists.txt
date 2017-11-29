
set(SOURCE "main.cpp"
                           "how_to_part_01_images.cpp"
                           "how_to_part_02_detections.cpp"
)

set(kwiver_examples_libraries vital_algo
                              kwiver_algo_ocv
                              kwiver_algo_vxl)

if(fletch_ENABLED_YAMLCPP)
  # YAML parsing and raw kpf examples
  find_package( yaml-cpp REQUIRED )
  list(APPEND SOURCE kpf/kpf_example.cxx
                     kpf/kpf_example_simple.cxx
                     kpf/kpf_example_complex.cxx
                     kpf/kpf_text_reader.cxx
                     kpf/kpf_yaml_reader.cxx
                     kpf/yaml_parser.cxx
                     kpf/kpf_example_yaml.cxx
  )

  list(APPEND kwiver_examples_libraries kwiver_algo_kpf kpf_yaml ${YAML_CPP_LIBRARIES})
  if(WIN32)
    set(kwiver_examples_flags YAML_CPP_DLL)
  endif()
endif()

add_executable(kwiver_examples ${SOURCE})
target_link_libraries(kwiver_examples ${kwiver_examples_libraries})
target_compile_definitions(kwiver_examples PRIVATE ${kwiver_examples_flags})
