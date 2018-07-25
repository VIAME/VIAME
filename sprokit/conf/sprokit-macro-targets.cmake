# Target management functions for the sprokit project.
# The following functions are defined:
#
#   sprokit_install
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
