project(vistk_processes_image)

set(image_srcs
  grayscale_process.cxx
  image_reader_process.cxx
  registration.cxx
  vil_helper.cxx)

set(image_reader_process_build_options
  ${boost_filesystem_options})

set_source_files_properties(image_reader_process.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${image_reader_process_build_options}")

set(vil_helper_process_build_options
  ${boost_filesystem_options})

set_source_files_properties(vil_helper_process.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${vil_helper_process_build_options}")

add_vistk_plugin(processes_image
  MAKE_VISTK_PROCESSES_IMAGE_LIB
  ${image_srcs})
target_link_libraries(processes_image
  vistk_pipeline_types
  vistk_pipeline
  vil
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
