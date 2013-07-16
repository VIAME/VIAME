# Target management functions for the sprokit project.
# The following functions are defined:
#
#   sprokit_install
#   sprokit_add_executable
#   sprokit_add_library
#   sprokit_add_plugin
#   sprokit_private_header_group
#   sprokit_private_template_group
#   sprokit_install_headers
#   sprokit_install_pipelines
#   sprokit_install_clusters
#   sprokit_install_includes
#   sprokit_add_helper_library
#
# The following variables may be used to control the behavior of the functions:
#
#   export_name
#     The export target name for installed targets. This is the name used for
#     install(EXPORT ...) calls
#
#   no_export
#     If set, the target will not be exported. See the CMake documentation for
#     what exporting a target means.
#
#   no_install
#     If set, the target will not be installed.
#
#   library_subdir
#     If set, targets will be placed into the directory with this as a suffix.
#     This is necessary due to the way that systems which use the
#     CMAKE_BUILD_TYPE as a directory in the output.
#
#   library_subdir_suffix
#     If set, the suffix will be appended to the subdirectory for the target.
#     This is placed after the CMAKE_BUILD_TYPE subdirectory if necessary.
#
#   component
#     If set, the target will not be installed under this component (the
#     default is 'runtime').
#
# Their syntax is:
#
#   sprokit_export_targets(file [APPEND])
#     Write target exports to 'file'.
#
#   sprokit_install([args])
#     A wrapper around the install call which recognizes the 'no_install'
#     variable.
#
#   sprokit_add_executable(name [source ...])
#     Creates an executable that is built into the correct directory and is
#     installed.
#
#   sprokit_add_library(name [source ...])
#     Creates a library named that is built into the correct directory and
#     is installed. Additionally, the 'library_subdir' variable can be set to
#     put the library in the correct place on DLL systems (see the CMake
#     documentation on LIBRARY_OUTPUT_DIRECTORY).
#
#   sprokit_add_plugin(name define [source ...])
#     Creates a plugin for sprokit and installs it to the correct directory.
#
#   sprokit_private_header_group([source ...])
#   sprokit_private_template_group([source ...])
#     Add 'sources' to a subdirectory within IDEs which display sources for
#     each target. Useful for separating installed files from private files in
#     the UI.
#
#   sprokit_install_headers(subdir [header ...])
#     Installs the headers stored in the variable under a subdirectory. Headers
#     are always installed under the 'development' component.
#
#   sprokit_install_pipelines([pipeline ...])
#   sprokit_install_clusters([cluster ...])
#   sprokit_install_includes([include ...])
#     Install pipeline files into the correct
#     location.
#
#   sprokit_add_helper_library(name sourcevar [library ...])
#     Adds a static library which contains code shared between separate
#     libraries. The library is neither exported nor installed. The 'sourcevar'
#     argument is a list of source files for the library.

set(__sprokit_export_targets
  CACHE INTERNAL "Targets exported by sprokit")
set(sprokit_libraries
  CACHE INTERNAL "Shared libraries built as part of sprokit")

function (_sprokit_compile_pic name)
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

macro (_sprokit_export name)
  set(exports)

  if (NOT no_export)
    set(exports
      EXPORT ${export_name})

    set(__sprokit_export_targets
      ${__sprokit_export_targets}
      ${name}
      CACHE INTERNAL "Targets exported by sprokit")
  endif ()
endmacro ()

function (sprokit_export_targets file)
  export(
    TARGETS ${__sprokit_export_targets}
    ${ARGN}
    FILE    "${file}")
endfunction ()

function (sprokit_install)
  if (NOT no_install)
    install(${ARGN})
  endif ()
endfunction ()

function (sprokit_add_executable name)
  add_executable(${name}
    ${ARGN})
  set_target_properties(${name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${sprokit_output_dir}/bin")

  if (NOT component)
    set(component
      runtime)
  endif ()

  _sprokit_export(${name})

  sprokit_install(
    TARGETS     ${name}
    ${exports}
    DESTINATION bin
    COMPONENT   ${component})
endfunction ()

function (sprokit_add_library name)
  add_library("${name}"
    ${ARGN})
  set_target_properties("${name}"
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${sprokit_output_dir}/lib${library_subdir}${library_subdir_suffix}"
      LIBRARY_OUTPUT_DIRECTORY "${sprokit_output_dir}/lib${library_subdir}${library_subdir_suffix}"
      RUNTIME_OUTPUT_DIRECTORY "${sprokit_output_dir}/bin${library_subdir}${library_subdir_suffix}")

  add_dependencies("${name}"
    configure-config.h)

  foreach (config IN LISTS CMAKE_CONFIGURATION_TYPES)
    set(subdir "${library_subdir}${library_subdir_suffix}")

    if (CMAKE_CONFIGURATION_TYPES)
      set(subdir "${library_subdir}/${config}${library_subdir_suffix}")
    endif ()

    string(TOUPPER "${config}" upper_config)

    set_target_properties(${name}
      PROPERTIES
        "ARCHIVE_OUTPUT_DIRECTORY_${upper_config}" "${sprokit_output_dir}/lib${subdir}"
        "LIBRARY_OUTPUT_DIRECTORY_${upper_config}" "${sprokit_output_dir}/lib${subdir}"
        "RUNTIME_OUTPUT_DIRECTORY_${upper_config}" "${sprokit_output_dir}/bin${subdir}")
  endforeach ()

  if (NOT component)
    set(component
      runtime)
  endif ()

  get_target_property(target_type
    "${name}" TYPE)

  if (target_type STREQUAL "STATIC_LIBRARY")
    _sprokit_compile_pic("${name}")
  else ()
    set(sprokit_libraries
      ${sprokit_libraries}
      "${name}"
      CACHE INTERNAL "Libraries built as part of sprokit")
  endif ()

  _sprokit_export("${name}")

  sprokit_install(
    TARGETS       "${name}"
    ${exports}
    ARCHIVE
      DESTINATION "lib${LIB_SUFFIX}${library_subdir}"
    LIBRARY
      DESTINATION "lib${LIB_SUFFIX}${library_subdir}"
    RUNTIME
      DESTINATION "bin${library_subdir}"
    COMPONENT     ${component})
endfunction ()

function (sprokit_add_plugin name define)
  set(library_subdir /sprokit)

  set(no_export ON)

  sprokit_add_library(${name} MODULE
    ${ARGN})

  set_target_properties(${name}
    PROPERTIES
      DEFINE_SYMBOL    ${define}
      PREFIX           ""
      SUFFIX           ${CMAKE_SHARED_MODULE_SUFFIX})
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

function (sprokit_install_pipelines)
  sprokit_install(
    FILES       ${ARGN}
    DESTINATION share/sprokit/pipelines
    COMPONENT   pipeline)
endfunction ()

function (sprokit_install_clusters)
  sprokit_install(
    FILES       ${ARGN}
    DESTINATION share/sprokit/pipelines/clusters
    COMPONENT   pipeline)
endfunction ()

function (sprokit_install_includes)
  sprokit_install(
    FILES       ${ARGN}
    DESTINATION share/sprokit/pipelines/include
    COMPONENT   pipeline)
endfunction ()

function (sprokit_add_helper_library name sources)
  add_library(${name} STATIC
    ${${sources}})
  target_link_libraries(${name}
    LINK_PRIVATE
      ${ARGN})

  _sprokit_compile_pic(${name})
endfunction ()
