# Target management functions for the sprokit project.
# The following functions are defined:
#
#   sprokit_install
#   sprokit_add_executable
#   sprokit_add_library
#   sprokit_private_header_group
#   sprokit_private_template_group
#   sprokit_install_pipelines
#   sprokit_install_clusters
#   sprokit_install_includes
#
# The following variables may be used to control the behavior of the functions:
#
#   kwiver_export_name
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
#   sprokit_output_dir
#     The base directory to output all targets into.
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
#   sprokit_private_header_group([source ...])
#   sprokit_private_template_group([source ...])
#     Add 'sources' to a subdirectory within IDEs which display sources for
#     each target. Useful for separating installed files from private files in
#     the UI.
#
#   sprokit_install_pipelines([pipeline ...])
#   sprokit_install_clusters([cluster ...])
#   sprokit_install_includes([include ...])
#     Install pipeline files into the correct
#     location.
#

###
#
function (_sprokit_export name)
  set(exports
    PARENT_SCOPE)

  if (no_export)
    return()
  endif ()

  set(exports
    EXPORT ${kwiver_export_name}
    PARENT_SCOPE)

  set_property(GLOBAL APPEND
    PROPERTY kwiver_export_targets
    ${name})
endfunction ()

###
#
function (sprokit_export_targets file)
  get_property(sprokit_exports GLOBAL
    PROPERTY kwiver_export_targets)
  export(
    TARGETS ${sprokit_exports}
    ${ARGN}
    FILE    "${file}")
endfunction ()

###
#
function (sprokit_install)
  if (no_install)
    return()
  endif ()

  install(${ARGN})
endfunction ()

###
#
function (sprokit_private_header_group)
  source_group("Header Files\\Private"
    FILES ${ARGN})
endfunction ()

###
#
function (sprokit_private_template_group)
  source_group("Template Files\\Private"
    FILES ${ARGN})
endfunction ()

###
#
function (sprokit_install_pipelines)
  sprokit_install(
    FILES       ${ARGN}
    DESTINATION share/sprokit/pipelines
    COMPONENT   pipeline)
endfunction ()

###
# for cluster definitions
function (sprokit_install_clusters)
  sprokit_install(
    FILES       ${ARGN}
    DESTINATION share/sprokit/pipelines/clusters
    COMPONENT   pipeline)
endfunction ()

###
# for pipeline fragment files that are included
function (sprokit_install_includes)
  sprokit_install(
    FILES       ${ARGN}
    DESTINATION share/sprokit/pipelines/include
    COMPONENT   pipeline)
endfunction ()
