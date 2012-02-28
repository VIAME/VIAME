project(vistk_conf)

if (VISTK_ENABLE_TESTING)
  # Configure the warning and code coverage suppression file
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in"
    "${CMAKE_BINARY_DIR}/CTestCustom.cmake"
    COPYONLY)
endif (VISTK_ENABLE_TESTING)

set(vistk_config_file "${CMAKE_BINARY_DIR}/vistk-config.cmake")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-config.cmake.in"
  "${vistk_config_file}"
  @ONLY)

set(vistk_config_install_file "${CMAKE_BINARY_DIR}/vistk-config-install.cmake")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-config-install.cmake.in"
  "${vistk_config_install_file}"
  @ONLY)

set(vistk_macros_file "${CMAKE_BINARY_DIR}/vistk-macros.cmake")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-macros.cmake.in"
  "${vistk_macros_file}"
  @ONLY)

# Configure the CMake EXPORT file during installation
install(
  FILES       "${vistk_config_install_file}"
              "${vistk_macros_file}"
  DESTINATION "${LIBRARY_INSTALL_PATH}/cmake"
  COMPONENT   development)
install(
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}/cmake"
  FILE        vistk-config-targets.cmake
  COMPONENT   development)

if (UNIX)
  set(vistk_pkgconfig_file "${CMAKE_BINARY_DIR}/vistk.pc")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/vistk.pc.in"
    "${vistk_pkgconfig_file}"
    @ONLY)

  install(
    FILES       "${vistk_pkgconfig_file}"
    DESTINATION "${LIBRARY_INSTALL_PATH}/pkgconfig"
    COMPONENT   development)
endif (UNIX)
