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

set(image_helper_headers
  crop.h
  format.h
  grayscale.h
  macros.h
  operators.h
  pixtypes.h
  read.h
  warp.h
  write.h)

set(image_helper_all_srcs
  ${image_helper_srcs}
  ${image_helper_headers})

vistk_add_helper_library_sources(image_helper image_helper_all_srcs
  vistk_image
  vistk_pipeline
  vil)
