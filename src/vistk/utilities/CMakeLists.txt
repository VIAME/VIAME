project(vistk_utilities)

set(utilities_srcs
  timestamp.cxx
  timestamp_debug.cxx)

set(utilities_headers
  path.h
  timestamp.h
  timestamp_debug.h
  utilities-config.h)

add_library(vistk_utilities
  ${utilities_srcs})
target_link_libraries(vistk_utilities)
target_link_libraries(vistk_utilities LINK_INTERFACE_LIBRARIES)
set_target_properties(vistk_utilities
  PROPERTIES
    VERSION       ${vistk_version}
    SOVERSION     0
    DEFINE_SYMBOL MAKE_VISTK_UTILITIES_LIB)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    utilities)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${utilities_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/utilities"
  COMPONENT   runtime)
install(
  TARGETS     vistk_utilities
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}"
  COMPONENT   runtime)
