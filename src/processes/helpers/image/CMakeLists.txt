project(vistk_process_helpers_image)

set(image_helper_srcs
  crop.cxx
  format.cxx
  grayscale.cxx
  pixtypes.cxx
  read.cxx
  write.cxx)

add_library(image_helper STATIC
  ${image_helper_srcs})
target_link_libraries(image_helper
  vil)

if (NOT WIN32)
  set_target_properties(image_helper
    PROPERTIES
      COMPILE_FLAGS "-fPIC")
endif (NOT WIN32)
