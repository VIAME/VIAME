# The imported kwiver sources
#
# Phase 5 moved kwiver's vital, sprokit and python package into VIAME's own
# `library/` and `python/` trees, under the include prefix `viame/`. This sets
# up the roots they are reached through and the two macros that build them,
# for VIAME's directories and for what is left of `packages/kwiver` alike --
# so it has to be included before either is descended into.

# There can be only one vital in a process: identical code with identical
# guards compiles, links and runs, and then does not exit -- see
# design/lite-findings.md 1.1.
set( VIAME_LITE_ROOT_DIR    "${CMAKE_CURRENT_SOURCE_DIR}" )
set( VIAME_LITE_LIBRARY_DIR "${VIAME_LITE_ROOT_DIR}/library" )
set( VIAME_LITE_PYTHON_DIR  "${VIAME_LITE_ROOT_DIR}/python" )

set( VIAME_LITE_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/library-include" )
set( VIAME_LITE_GENERATED_DIR      "${CMAKE_CURRENT_BINARY_DIR}/library-generated" )

file( MAKE_DIRECTORY "${VIAME_LITE_SOURCE_INCLUDE_DIR}" )
file( MAKE_DIRECTORY "${VIAME_LITE_GENERATED_DIR}" )

# A link rather than a copy, so editing a header does not mean re-running
# CMake. Generated headers go to their own root beside it.
if( NOT EXISTS "${VIAME_LITE_SOURCE_INCLUDE_DIR}/viame" )
  file( CREATE_LINK "${VIAME_LITE_LIBRARY_DIR}"
                    "${VIAME_LITE_SOURCE_INCLUDE_DIR}/viame" SYMBOLIC
        RESULT _viame_lite_link )

  # Windows grants the symlink privilege to an administrator or a machine in
  # developer mode and to nobody else, so the call above fails on an ordinary
  # account. A directory junction needs no privilege and the compiler follows
  # it the same way, so it keeps the property the symlink was chosen for.
  if( NOT "${_viame_lite_link}" STREQUAL "0" AND WIN32 )
    file( TO_NATIVE_PATH "${VIAME_LITE_LIBRARY_DIR}" _viame_lite_target )
    file( TO_NATIVE_PATH "${VIAME_LITE_SOURCE_INCLUDE_DIR}/viame"
          _viame_lite_junction )
    execute_process(
      COMMAND cmd /c mklink /J "${_viame_lite_junction}" "${_viame_lite_target}"
      RESULT_VARIABLE _viame_lite_link
      OUTPUT_QUIET ERROR_QUIET )
  endif()

  # Last resort. It builds, but a header edited under `library/` is not seen
  # until CMake runs again, so say so -- a stale header otherwise reads as a
  # compiler that has lost its mind.
  if( NOT "${_viame_lite_link}" STREQUAL "0" )
    message( WARNING
      "Could not link ${VIAME_LITE_SOURCE_INCLUDE_DIR}/viame to "
      "${VIAME_LITE_LIBRARY_DIR}; copying instead. A header edited under "
      "library/ will need CMake to be re-run before the build sees it." )
    file( COPY "${VIAME_LITE_LIBRARY_DIR}/"
          DESTINATION "${VIAME_LITE_SOURCE_INCLUDE_DIR}/viame" )
  endif()
endif()

# The bindings include each other as `<python/kwiver/...>`, which used to
# resolve against kwiver's own source root and now resolves against this one.
include_directories( SYSTEM "${VIAME_LITE_ROOT_DIR}" )

# SYSTEM, because kwiver builds with -Werror=non-virtual-dtor and vital has a
# couple of headers that trip it.
include_directories( SYSTEM "${VIAME_LITE_SOURCE_INCLUDE_DIR}" )
include_directories( SYSTEM "${VIAME_LITE_GENERATED_DIR}" )
include_directories( SYSTEM "${VIAME_LITE_ROOT_DIR}/library/tpl/rapidjson" )
include_directories( SYSTEM "${VIAME_LITE_ROOT_DIR}/library/tpl/cxxopts" )

# Rebase a list of file names onto the imported tree.
#
# Kwiver's lists are the superset: the import took what
# design/lite-kwiver-files.txt says VIAME reaches, and the rest -- readers for
# formats VIAME does not read, interfaces nothing implements -- was left
# behind for P5-T06 to have pruned anyway. Anything that did not come across
# is dropped here and reported, so the difference is visible rather than
# silent. Absolute paths are generated files and are left alone.
macro( viame_lite_rebase list_var subdir )
  set( _viame_lite_rebased )
  set( _viame_lite_dropped )

  foreach( _viame_lite_file IN LISTS ${list_var} )
    if( IS_ABSOLUTE "${_viame_lite_file}" )
      list( APPEND _viame_lite_rebased "${_viame_lite_file}" )
    else()
      set( _viame_lite_path
        "${VIAME_LITE_LIBRARY_DIR}/${subdir}/${_viame_lite_file}" )

      if( EXISTS "${_viame_lite_path}" )
        list( APPEND _viame_lite_rebased "${_viame_lite_path}" )
      else()
        list( APPEND _viame_lite_dropped "${_viame_lite_file}" )
      endif()
    endif()
  endforeach()

  if( _viame_lite_dropped )
    list( LENGTH _viame_lite_dropped _viame_lite_count )
    message( STATUS
      "vital/${subdir}: ${_viame_lite_count} file(s) not imported: ${_viame_lite_dropped}" )
  endif()

  set( ${list_var} ${_viame_lite_rebased} )
endmacro()

# Republish a generated header under the prefix the imported sources use.
macro( viame_lite_generated name subdir )
  configure_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${name}"
    "${VIAME_LITE_GENERATED_DIR}/viame/${subdir}/${name}" COPYONLY )
  viame_install_headers(
    "${VIAME_LITE_GENERATED_DIR}/viame/${subdir}/${name}"
    SUBDIR viame/${subdir}
    NOPATH )
endmacro()

