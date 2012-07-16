project(vistk_image)

set(image_srcs
  warp_image.cxx)

set(image_headers
  image-config.h
  warp_image.h)

vistk_add_library(vistk_image
  ${image_srcs})
target_link_libraries(vistk_image
  vistk_utilities
  vil
  vgl)
target_link_libraries(vistk_image LINK_INTERFACE_LIBRARIES
  vistk_utilities
  vil)
set_target_properties(vistk_image
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_IMAGE_LIB)

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  image)

vistk_install_headers(vistk/image ${image_headers})

vistk_configure_pkgconfig(vistk-image)
