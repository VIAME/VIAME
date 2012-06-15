# Target management functions for the vistk project.
# The following functions are defined:
#   vistk_install
#   vistk_add_executable
#   vistk_add_library
#   vistk_install_headers
# Their syntax is:
#   vistk_install([args])
#     A wrapper around the install call which can be suppressed by setting the
#     'suppress_install' variable to a truthy value. Used in the other macros
#     for use when the product should not be installed.
#   vistk_add_executable(name [source ...])
#     Creates an executable that is built into the correct directory and is
#     installed. The 'component' variable can be set to override the default of
#     installing with the 'runtime' component.
#   vistk_add_library(name [source ...])
#     Creates a library named that is built into the correct directory and
#     is installed. The 'component' variable can be set to override the default
#     of installing with the 'runtime' component. Additionally, the
#     'library_subdir' variable can be set to put the library in the correct
#     place on DLL systems (see the CMake documentation on
#     LIBRARY_OUTPUT_DIRECTORY).
#   vistk_install_headers(subdir [headers ...])
#     Installs the headers stored in the variable under a subdirectory.

function (vistk_install)
  if (NOT suppress_install)
    install(${ARGN})
  endif (NOT suppress_install)
endfunction (vistk_install)

function (vistk_add_executable name)
  add_executable(${name}
    ${ARGN})
  set_target_properties(${name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${vistk_binary_dir}/bin")

  if ("${component}" STREQUAL "")
    set(component
      runtime)
  endif ("${component}" STREQUAL "")

  vistk_install(
    TARGETS     ${name}
    EXPORT      vistk_exports
    DESTINATION bin
    COMPONENT   ${component})
endfunction (vistk_add_executable)

function (vistk_add_library name)
  add_library(${name}
    ${ARGN})
  set_target_properties(${name}
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${vistk_binary_dir}/lib${library_subdir}"
      LIBRARY_OUTPUT_DIRECTORY "${vistk_binary_dir}/lib${library_subdir}"
      RUNTIME_OUTPUT_DIRECTORY "${vistk_binary_dir}/bin${library_subdir}")

  add_dependencies(${name}
    configure-config.h)

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    set(subdir ${library_subdir})

    if (WIN32)
      set(subdir "/${config}${subdir}")
    endif (WIN32)

    string(TOUPPER "${config}" upper_config)

    set_target_properties(${name}
      PROPERTIES
        "ARCHIVE_OUTPUT_DIRECTORY_${upper_config}" "${vistk_binary_dir}/lib${subdir}"
        "LIBRARY_OUTPUT_DIRECTORY_${upper_config}" "${vistk_binary_dir}/lib${subdir}"
        "RUNTIME_OUTPUT_DIRECTORY_${upper_config}" "${vistk_binary_dir}/bin${subdir}")
  endforeach (config)

  if ("${component}" STREQUAL "")
    set(component
      runtime)
  endif ("${component}" STREQUAL "")

  set(exports)

  if ("${library_subdir}" STREQUAL "")
    set(exports
      EXPORT vistk_exports)
  endif ("${library_subdir}" STREQUAL "")

  vistk_install(
    TARGETS       ${name}
    ${exports}
    ARCHIVE
      DESTINATION "lib${LIB_SUFFIX}${library_subdir}"
    LIBRARY
      DESTINATION "lib${LIB_SUFFIX}${library_subdir}"
    RUNTIME
      DESTINATION bin
    COMPONENT     ${component})
endfunction (vistk_add_library)

function (vistk_install_headers subdir)
  vistk_install(
    FILES       ${ARGN}
    DESTINATION "include/${subdir}"
    COMPONENT   development)
endfunction (vistk_install_headers)
