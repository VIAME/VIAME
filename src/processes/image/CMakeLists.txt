project(vistk_processes_image)

set(image_srcs
  grayscale_process.cxx
  registration.cxx)

add_vistk_plugin(processes_image
  MAKE_VISTK_PROCESSES_IMAGE_LIB
  ${image_srcs})
target_link_libraries(processes_image
  vistk_pipeline_types
  vistk_pipeline
  vil)
