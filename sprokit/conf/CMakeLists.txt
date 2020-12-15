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
  if(SKBUILD)
    return()
  endif()
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
if(NOT SKBUILD)
sprokit_install(
  FILES       "${sprokit_macros_file}"
  DESTINATION "${cmakedir}"
  COMPONENT   development)
endif()
## sprokit_configure_pkgconfig(sprokit)
