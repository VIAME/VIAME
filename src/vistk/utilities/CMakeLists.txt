project(vistk_utilities)

set(utilities_headers
  path.h
  utilities-config.h)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    utilities)
endif (VISTK_ENABLE_DOCUMENTATION)

install(
  FILES       ${utilities_headers}
  DESTINATION "${INCLUDE_INSTALL_PATH}/vistk/utilities"
  COMPONENT   runtime)
