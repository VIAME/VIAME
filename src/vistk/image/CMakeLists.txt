project(vistk_image)

set(image_srcs
  warp_image.cxx)

set(image_templates
  warp_image.txx)

set(image_headers
  image-config.h
  warp_image.h)

set(image_private_headers
  instantiate.h)

vistk_private_header_group(${image_private_headers})
vistk_add_library(vistk_image
  ${image_srcs}
  ${image_templates}
  ${image_headers}
  ${image_private_headers})
target_link_libraries(vistk_image
  LINK_PUBLIC
    vistk_utilities
    vil
  LINK_PRIVATE
    vgl)
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
