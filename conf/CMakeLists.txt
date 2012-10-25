project(vistk_conf)

if (VISTK_ENABLE_TESTING)
  # Configure the warning and code coverage suppression file
  vistk_configure_file(CTestCustom.cmake
    "${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in"
    "${vistk_binary_dir}/CTestCustom.cmake")
endif ()

if (VISTK_ENABLE_CDASH)
  find_program(GCOV_EXECUTABLE gcov)
  execute_process(
    COMMAND         hostname
    OUTPUT_VARIABLE default_site_name
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX REPLACE "^([^.]*)(\..*)?$" "\1" default_site_name "${default_site_name}")
  set(CDASH_SITE_NAME "${default_site_name}"
    CACHE STRING "The hostname to submit to the dashboard with")
  set(CDASH_BUILD_NAME ""
    CACHE STRING "String describing the build (\"regex\": 'os-os_version-arch-buildtool-compiler-compiler_version(-dep-depversion)*(-feature)?')")

  # Configure dashboard script
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/dashboard.ctest.in"
    "${vistk_binary_dir}/dashboard.ctest"
    @ONLY)
endif ()

set(vistk_config_file "${vistk_binary_dir}/vistk-config.cmake")
vistk_configure_file(vistk-config.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-config.cmake.in"
  "${vistk_config_file}"
  vistk_source_dir
  vistk_binary_dir
  vistk_version
  VISTK_ENABLE_PYTHON
  VISTK_ENABLE_PYTHON3
  PYTHON_EXECUTABLE
  BUILD_SHARED_LIBS)

set(vistk_config_install_file "${vistk_binary_dir}/vistk-config-install.cmake")
vistk_configure_file(vistk-config-install.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-config-install.cmake.in"
  "${vistk_config_install_file}"
  CMAKE_INSTALL_PREFIX
  LIB_SUFFIX
  vistk_version
  VISTK_ENABLE_PYTHON
  VISTK_ENABLE_PYTHON3
  PYTHON_EXECUTABLE
  BUILD_SHARED_LIBS)

if (WIN32)
  set(cmakedir cmake)
else ()
  set(cmakedir lib${LIB_SUFFIX}/cmake/vistk)
endif ()

set(vistk_macros_file "${vistk_binary_dir}/vistk-macros.cmake")
vistk_configure_file(vistk-macros.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/vistk-macros.cmake.in"
  "${vistk_macros_file}")

# Configure the CMake EXPORT file during installation
install(
  FILES       "${vistk_macros_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development)
install(
  FILES       "${vistk_config_install_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development
  RENAME      vistk-config.cmake)
install(
  EXPORT      vistk_exports
  DESTINATION "${cmakedir}"
  FILE        vistk-config-targets.cmake
  COMPONENT   development)

vistk_configure_pkgconfig(vistk)
