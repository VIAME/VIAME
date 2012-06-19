project(vistk_conf)

if (VISTK_ENABLE_TESTING)
  # Configure the warning and code coverage suppression file
  vistk_configure_file(CTestCustom.cmake
    "${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in"
    "${vistk_binary_dir}/CTestCustom.cmake")
endif (VISTK_ENABLE_TESTING)

set(vistk_config_file "${vistk_binary_dir}/vistk-config.cmake")
vistk_configure_file(vistk-config.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-config.cmake.in"
  "${vistk_config_file}"
  vistk_source_dir
  vistk_binary_dir
  vistk_version
  VISTK_ENABLE_PYTHON
  VISTK_ENABLE_PYTHON3
  PYTHON_EXECUTABLE)

set(vistk_config_install_file "${vistk_binary_dir}/vistk-config-install.cmake")
vistk_configure_file(vistk-config-install.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-config-install.cmake.in"
  "${vistk_config_install_file}"
  CMAKE_INSTALL_PREFIX
  LIB_SUFFIX
  vistk_version
  VISTK_ENABLE_PYTHON
  VISTK_ENABLE_PYTHON3
  PYTHON_EXECUTABLE)

if (WIN32)
  set(cmakedir cmake)
else (WIN32)
  set(cmakedir lib${LIB_SUFFIX}/cmake/vistk)
endif (WIN32)

set(vistk_macros_file "${vistk_binary_dir}/vistk-macros.cmake")
vistk_configure_file(vistk-macros.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-macros.cmake.in"
  "${vistk_macros_file}")

# Configure the CMake EXPORT file during installation
install(
  FILES       "${vistk_config_install_file}"
              "${vistk_macros_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development)
install(
  EXPORT      vistk_exports
  DESTINATION "${cmakedir}"
  FILE        vistk-config-targets.cmake
  COMPONENT   development)

vistk_configure_pkgconfig(vistk)
