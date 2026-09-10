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
                    "${VIAME_LITE_SOURCE_INCLUDE_DIR}/viame" SYMBOLIC )
endif()

# The bindings include each other as `<python/kwiver/...>`, which used to
# resolve against kwiver's own source root and now resolves against this one.
include_directories( SYSTEM "${VIAME_LITE_ROOT_DIR}" )

# SYSTEM, because kwiver builds with -Werror=non-virtual-dtor and vital has a
# couple of headers that trip it.
include_directories( SYSTEM "${VIAME_LITE_SOURCE_INCLUDE_DIR}" )
include_directories( SYSTEM "${VIAME_LITE_GENERATED_DIR}" )
include_directories( SYSTEM "${VIAME_LITE_ROOT_DIR}/third_party/cereal" )
# cereal ships rapidjson under its own external/; kwiver includes it directly
include_directories( SYSTEM
  "${VIAME_LITE_ROOT_DIR}/third_party/cereal/cereal/external" )
include_directories( SYSTEM "${VIAME_LITE_ROOT_DIR}/third_party/cxxopts" )

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
  kwiver_install_headers(
    "${VIAME_LITE_GENERATED_DIR}/viame/${subdir}/${name}"
    SUBDIR viame/${subdir}
    NOPATH )
endmacro()

