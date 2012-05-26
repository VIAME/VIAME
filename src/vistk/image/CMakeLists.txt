project(vistk_image)

set(image_srcs
  warp_image.cxx)

set(image_headers
  image-config.h
  warp_image.h)

add_library(vistk_image
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

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    image)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${image_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/image"
  COMPONENT   development)
install(
  TARGETS     vistk_image
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)

vistk_configure_pkgconfig(vistk-image)
