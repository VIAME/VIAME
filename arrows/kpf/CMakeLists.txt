find_package( yaml-cpp REQUIRED )

set(CMAKE_FOLDER "Arrows/KPF")

add_subdirectory(yaml)

# Build/install plugin for kpf support for vital objects
#

set( headers
  detected_object_set_input_kpf.h
  detected_object_set_output_kpf.h
)

set( sources
  vital_kpf_adapters.h
  detected_object_set_input_kpf.cxx
  detected_object_set_output_kpf.cxx
  )

set( kpf_linked_libs
  kwiversys)


kwiver_install_headers(
  SUBDIR     arrows/kpf
  ${headers}
  )
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_kpf_export.h
  NOPATH   SUBDIR     arrows/kpf
  )


kwiver_add_library( kwiver_algo_kpf
  ${headers}
  ${sources}
  )

target_link_libraries( kwiver_algo_kpf
  PUBLIC               vital_algo kpf_yaml
  PRIVATE              ${YAML_CPP_LIBRARIES}
  )

algorithms_create_plugin( kwiver_algo_kpf
  register_algorithms.cxx
  )

# Python Support?
#if( KWIVER_ENABLE_PYTHON )
#  kwiver_create_python_init( kwiver/arrows/kpf )
#
#  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/generate_headers.py
#    kwiver/arrows/kpf
#    generate_headers )
#endif()
