project(vistk_utilities)

set(utilities_srcs
  homography.cxx
  homography_debug.cxx
  plane_ref.cxx
  plane_ref_debug.cxx
  timestamp.cxx
  timestamp_debug.cxx
  utm.cxx
  utm_debug.cxx)

set(utilities_headers
  homography.h
  homography_debug.h
  homography_types.h
  path.h
  plane_ref.h
  plane_ref_debug.h
  timestamp.h
  timestamp_debug.h
  utilities-config.h
  utm.h
  utm_debug.h)

add_library(vistk_utilities
  ${utilities_srcs})
target_link_libraries(vistk_utilities
  vgl_algo)
target_link_libraries(vistk_utilities LINK_INTERFACE_LIBRARIES
  vgl_algo)
set_target_properties(vistk_utilities
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_UTILITIES_LIB)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    utilities)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${utilities_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/utilities"
  COMPONENT   development)
install(
  TARGETS     vistk_utilities
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)

if (UNIX)
  set(vistk_utilities_pkgconfig_file "${vistk_binary_dir}/vistk-utilities.pc")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/vistk-utilities.pc.in"
    "${vistk_utilities_pkgconfig_file}"
    @ONLY)
  set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${vistk_utilities_pkgconfig_file}")

  install(
    FILES       "${vistk_utilities_pkgconfig_file}"
    DESTINATION "${LIBRARY_INSTALL_PATH}/pkgconfig"
    COMPONENT   development)
endif (UNIX)
