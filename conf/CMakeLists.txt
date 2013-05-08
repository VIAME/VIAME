project(sprokit_conf)

if (SPROKIT_ENABLE_TESTING)
  # Configure the warning and code coverage suppression file
  sprokit_configure_file(CTestCustom.cmake
    "${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake.in"
    "${sprokit_binary_dir}/CTestCustom.cmake")
endif ()

if (SPROKIT_ENABLE_CDASH)
  find_program(GCOV_EXECUTABLE gcov)
  execute_process(
    COMMAND         hostname
    OUTPUT_VARIABLE default_site_name
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REGEX REPLACE "^([^.]+)(\\..*)?$" "\\1" short_default_site_name "${default_site_name}")
  set(CDASH_SITE_NAME "${short_default_site_name}"
    CACHE STRING "The hostname to submit to the dashboard with")
  set(CDASH_BUILD_NAME ""
    CACHE STRING "String describing the build (\"regex\": 'os-os_version-arch-buildtool-compiler-compiler_version(-dep-depversion)*(-feature)?')")

  # Configure dashboard script
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/dashboard.ctest.in"
    "${sprokit_binary_dir}/dashboard.ctest"
    @ONLY)
endif ()

set(sprokit_config_file "${sprokit_binary_dir}/sprokit-config.cmake")
sprokit_configure_file(sprokit-config.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/sprokit-config.cmake.in"
  "${sprokit_config_file}"
  sprokit_source_dir
  sprokit_binary_dir
  sprokit_version
  SPROKIT_ENABLE_PYTHON
  SPROKIT_ENABLE_PYTHON3
  PYTHON_EXECUTABLE
  BUILD_SHARED_LIBS)

set(sprokit_config_install_file "${sprokit_binary_dir}/sprokit-config-install.cmake")
sprokit_configure_file(sprokit-config-install.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/sprokit-config-install.cmake.in"
  "${sprokit_config_install_file}"
  CMAKE_INSTALL_PREFIX
  LIB_SUFFIX
  sprokit_version
  SPROKIT_ENABLE_PYTHON
  SPROKIT_ENABLE_PYTHON3
  PYTHON_EXECUTABLE
  BUILD_SHARED_LIBS)

if (WIN32)
  set(cmakedir cmake)
else ()
  set(cmakedir lib${LIB_SUFFIX}/cmake/sprokit)
endif ()

set(sprokit_macros_file "${sprokit_binary_dir}/sprokit-macros.cmake")
file(WRITE "${sprokit_macros_file}"
"# Macros from sprokit.

")

set(macro_files
  configure
  fixes
  targets
  tests)

if (SPROKIT_ENABLE_PYTHON)
  set(macro_files
    ${macro_files}
    python)
endif ()

foreach (macro_file ${macro_files})
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/sprokit-macro-${macro_file}.cmake"
    "${sprokit_binary_dir}/sprokit-macro-${macro_file}.cmake"
    COPY_ONLY)
  install(
    FILES       "${sprokit_binary_dir}/sprokit-macro-${macro_file}.cmake"
    DESTINATION "${cmakedir}"
    COMPONENT   development)
  file(APPEND "${sprokit_macros_file}"
"include(\"${CMAKE_CURRENT_LIST_DIR}/sprokit-macro-${macro_file}.cmake\")
")
endforeach ()

# Configure the CMake EXPORT file during installation
install(
  FILES       "${sprokit_macros_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development)
install(
  FILES       "${sprokit_config_install_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development
  RENAME      sprokit-config.cmake)
sprokit_export_targets("${sprokit_binary_dir}/sprokit-config-targets.cmake")
install(
  EXPORT      ${export_name}
  DESTINATION "${cmakedir}"
  FILE        sprokit-config-targets.cmake
  COMPONENT   development)

sprokit_configure_pkgconfig(sprokit)
