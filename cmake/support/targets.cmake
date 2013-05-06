# Target management functions for the sprokit project.
# The following functions are defined:
#   sprokit_install
#   sprokit_add_executable
#   sprokit_add_library
#   sprokit_install_headers
# Their syntax is:
#   sprokit_install([args])
#     A wrapper around the install call which can be suppressed by setting the
#     'suppress_install' variable to a truthy value. Used in the other macros
#     for use when the product should not be installed.
#   sprokit_add_executable(name [source ...])
#     Creates an executable that is built into the correct directory and is
#     installed. The 'component' variable can be set to override the default of
#     installing with the 'runtime' component.
#   sprokit_add_library(name [source ...])
#     Creates a library named that is built into the correct directory and
#     is installed. The 'component' variable can be set to override the default
#     of installing with the 'runtime' component. Additionally, the
#     'library_subdir' variable can be set to put the library in the correct
#     place on DLL systems (see the CMake documentation on
#     LIBRARY_OUTPUT_DIRECTORY).
#   sprokit_install_headers(subdir [headers ...])
#     Installs the headers stored in the variable under a subdirectory.

set(sprokit_libraries CACHE INTERNAL "Libraries built as part of sprokit")

set(sprokit_export_file
  "${sprokit_binary_dir}/sprokit-config-targets.cmake")
export(
  TARGETS
  FILE    "${sprokit_export_file}")

function (sprokit_install)
  if (NOT suppress_install)
    install(${ARGN})
  endif ()
endfunction ()

macro (sprokit_export name)
  set(exports)

  if (NOT no_export)
    set(exports
      EXPORT sprokit_exports)

    export(
      TARGETS ${name}
      APPEND
      FILE    "${sprokit_export_file}")
  endif ()
endmacro ()

function (sprokit_compile_pic name)
  # TODO: Bump minimum CMake version to 2.8.9
  if (CMAKE_VERSION VERSION_GREATER "2.8.9")
    set_target_properties(${name}
      PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)
  elseif (NOT MSVC)
    set_target_properties(${name}
      PROPERTIES
        COMPILE_FLAGS "-fPIC")
  endif ()
endfunction ()

function (sprokit_add_executable name)
  add_executable(${name}
    ${ARGN})
  set_target_properties(${name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${sprokit_binary_dir}/bin")

  if ("${component}" STREQUAL "")
    set(component
      runtime)
  endif ()

  sprokit_export(${name})

  sprokit_install(
    TARGETS     ${name}
    ${exports}
    DESTINATION bin
    COMPONENT   ${component})
endfunction ()

function (sprokit_add_library name)
  add_library(${name}
    ${ARGN})
  set_target_properties(${name}
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${sprokit_binary_dir}/lib${library_subdir}"
      LIBRARY_OUTPUT_DIRECTORY "${sprokit_binary_dir}/lib${library_subdir}"
      RUNTIME_OUTPUT_DIRECTORY "${sprokit_binary_dir}/bin${library_subdir}")

  add_dependencies(${name}
    configure-config.h)

  foreach (config ${CMAKE_CONFIGURATION_TYPES})
    set(subdir ${library_subdir})

    if (CMAKE_CONFIGURATION_TYPES)
      set(subdir "${subdir}/${config}")
    endif ()

    string(TOUPPER "${config}" upper_config)

    set_target_properties(${name}
      PROPERTIES
        "ARCHIVE_OUTPUT_DIRECTORY_${upper_config}" "${sprokit_binary_dir}/lib${subdir}"
        "LIBRARY_OUTPUT_DIRECTORY_${upper_config}" "${sprokit_binary_dir}/lib${subdir}"
        "RUNTIME_OUTPUT_DIRECTORY_${upper_config}" "${sprokit_binary_dir}/bin${subdir}")
  endforeach ()

  if ("${component}" STREQUAL "")
    set(component
      runtime)
  endif ()

  get_target_property(target_type
    ${name} TYPE)

  if (target_type STREQUAL "STATIC_LIBRARY")
    sprokit_compile_pic(${name})
  else ()
    set(sprokit_libraries
      ${sprokit_libraries}
      ${name}
      CACHE INTERNAL "Libraries built as part of sprokit")
  endif ()

  sprokit_export(${name})

  sprokit_install(
    TARGETS       ${name}
    ${exports}
    ARCHIVE
      DESTINATION "lib${LIB_SUFFIX}${library_subdir}"
    LIBRARY
      DESTINATION "lib${LIB_SUFFIX}${library_subdir}"
    RUNTIME
      DESTINATION "bin${library_subdir}"
    COMPONENT     ${component})
endfunction ()

function (sprokit_private_header_group)
  source_group("Header Files\\Private"
    FILES ${ARGN})
endfunction ()

function (sprokit_private_template_group)
  source_group("Template Files\\Private"
    FILES ${ARGN})
endfunction ()

function (sprokit_install_headers subdir)
  sprokit_install(
    FILES       ${ARGN}
    DESTINATION "include/${subdir}"
    COMPONENT   development)
endfunction ()

function (sprokit_add_helper_library_sources name sources)
  add_library(${name} STATIC
    ${${sources}})
  target_link_libraries(${name}
    LINK_PRIVATE
      ${ARGN})

  sprokit_compile_pic(${name})
endfunction ()

function (sprokit_add_helper_library name)
  set(helper_sources
    ${name}.cxx)

  sprokit_add_helper_library_sources(${name} helper_sources
    ${ARGN})
endfunction ()
