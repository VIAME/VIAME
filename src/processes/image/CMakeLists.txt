project(vistk_processes_image)

set(image_srcs
  combine_masks_process.cxx
  crop_image_process.cxx
  grayscale_process.cxx
  image_reader_process.cxx
  image_writer_process.cxx
  layered_image_reader_process.cxx
  registration.cxx
  video_reader_process.cxx
  warp_image_process.cxx)

add_vistk_plugin(processes_image
  MAKE_VISTK_PROCESSES_IMAGE_LIB
  ${image_srcs})
target_link_libraries(processes_image
  vistk_image
  vistk_utilities
  vistk_pipeline
  video_helper
  image_helper
  vgl
  vgl_algo
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
