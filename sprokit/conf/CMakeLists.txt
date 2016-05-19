project(sprokit_conf)

if (KWIVER_ENABLE_TESTS)
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
  "${CMAKE_CURRENT_SOURCE_DIR}/sprokit-config-common.cmake.in"
  "${sprokit_config_file}"
  KWIVER_VERSION
  Boost_MAJOR_VERSION
  Boost_MINOR_VERSION
  Boost_SUBMINOR_VERSION
  BOOST_MANGLE_NAMESPACE
  Boost_USE_STATIC_LIBS
  Boost_INCLUDE_DIR)

set(config_vars
  KWIVER_ENABLE_PYTHON
  SPROKIT_ENABLE_PYTHON3
  PYTHON_EXECUTABLE
  PYTHON_VERSION
  PYTHON_ABIFLAGS
  BUILD_SHARED_LIBS
  SPROKIT_ENABLE_CXX11)

set(sprokit_config_build_file "${sprokit_binary_dir}/sprokit-config-variables.cmake")
sprokit_configure_file(sprokit-config-build.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/sprokit-config-build.cmake.in"
  "${sprokit_config_build_file}"
  sprokit_source_dir
  sprokit_binary_dir
  sprokit_python_output_path
  ${config_vars})

set(sprokit_config_install_file "${sprokit_binary_dir}/sprokit-config-install.cmake")
sprokit_configure_file(sprokit-config-install.cmake
  "${CMAKE_CURRENT_SOURCE_DIR}/sprokit-config-install.cmake.in"
  "${sprokit_config_install_file}"
  KWIVER_VERSION
  CMAKE_INSTALL_PREFIX
  LIB_SUFFIX
  ${config_vars})

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

if (KWIVER_ENABLE_PYTHON)
  list(APPEND macro_files
    python)
endif ()

set(fixes_macro_vars
  sprokit_fixes)

function (sprokit_install_macro_file path)
  sprokit_install(
    FILES       "${path}"
    DESTINATION "${cmakedir}"
    COMPONENT   development)
  file(APPEND "${sprokit_macros_file}"
"include(\"${CMAKE_CURRENT_LIST_DIR}/sprokit-macro-${macro_file}.cmake\")
")
endfunction ()

foreach (macro_file IN LISTS macro_files)
  set(filename "sprokit-macro-${macro_file}.cmake")
  sprokit_configure_file(macro-${macro_file}
    "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
    "${sprokit_binary_dir}/${filename}"
    ${${macro_file}_macro_vars})
  sprokit_install_macro_file("${sprokit_binary_dir}/${filename}")
endforeach ()

foreach (macro_file IN LISTS configured_macro_files)
  sprokit_install_macro_file("${sprokit_binary_dir}/${filename}")
endforeach ()

# Configure the CMake EXPORT file during installation
sprokit_install(
  FILES       "${sprokit_config_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development
  RENAME      sprokit-config.cmake)
sprokit_install(
  FILES       "${sprokit_macros_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development)
sprokit_install(
  FILES       "${sprokit_config_install_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development
  RENAME      sprokit-config-variables.cmake)
sprokit_export_targets("${sprokit_binary_dir}/sprokit-config-targets.cmake")
sprokit_install(
  EXPORT      ${kwiver_export_name}
  DESTINATION "${cmakedir}"
  FILE        sprokit-config-targets.cmake
  COMPONENT   development)

## sprokit_configure_pkgconfig(sprokit)
