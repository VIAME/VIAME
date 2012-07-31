project(vistk_processes_helpers_image)

set(image_helper_srcs
  crop.cxx
  format.cxx
  grayscale.cxx
  operators.cxx
  pixtypes.cxx
  read.cxx
  warp.cxx
  write.cxx)

vistk_add_helper_library_sources(image_helper image_helper_srcs
  vistk_image
  vistk_pipeline
  vil)
