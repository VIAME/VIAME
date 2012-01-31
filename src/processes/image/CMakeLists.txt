project(vistk_processes_image)

set(image_srcs
  crop_image_process.cxx
  grayscale_process.cxx
  image_reader_process.cxx
  image_writer_process.cxx
  registration.cxx
  video_reader_process.cxx
  warp_image_process.cxx)

set(image_reader_process_build_options
  ${boost_filesystem_options})

set_source_files_properties(image_reader_process.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${image_reader_process_build_options}")

set(image_writer_process_build_options
  ${boost_filesystem_options})

set_source_files_properties(image_writer_process.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${image_writer_process_build_options}")

set(video_reader_process_build_options
  ${boost_filesystem_options})

set_source_files_properties(video_reader_process.cxx
  PROPERTIES
    COMPILE_DEFINITIONS "${video_reader_process_build_options}")

add_vistk_plugin(processes_image
  MAKE_VISTK_PROCESSES_IMAGE_LIB
  ${image_srcs})
target_link_libraries(processes_image
  vistk_image
  vistk_utilities
  vistk_pipeline
  video_helper
  image_helper
  vidl
  vgl
  vgl_algo
  vil
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
