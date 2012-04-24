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

add_library(image_helper STATIC
  ${image_helper_srcs})
target_link_libraries(image_helper
  vistk_image
  vistk_pipeline
  vil)

if (NOT WIN32)
  set_target_properties(image_helper
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)
