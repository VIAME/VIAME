# VIAME's python helpers
#
# These were kwiver's `kwiver-utils-python.cmake` until P8-T08, and the two
# things they do are put a `.py` file where the interpreter will find it and
# build a pybind11 extension module beside it.
#
# **The configure step is a copy.** kwiver ran every `.py` through
# `kwiver_configure_file`, which spawns `cmake -P` on a helper script to do a
# `configure_file` at build time so that `@VAR@` in a python source is
# substituted. No python module VIAME installs contains one: the only `@...@`
# in any `.py` under `python/`, `library/` or `plugins/` is `@template@`, in
# `plugins/templates/python/template_detector.py`, which is a template for
# users rather than a module that is built. So it is `copy_if_different`,
# which is one process instead of two and does nothing when nothing changed.
#
# It stays a **build-time** copy. Editing a python file and running `make` is
# expected to update the build tree; `configure_file` at configure time would
# need cmake re-run, which is the kind of change people discover by losing an
# afternoon.
#
# **`SKBUILD` and `python_noarch` are gone.** VIAME is not built through
# scikit-build and never sets either. `VIAME_PYTHON_SYMLINK` stays -- it is a
# VIAME option, off by default, and the reason it exists is the same as the
# reason the copy is build-time.

include_guard( GLOBAL )

# Everything that has to happen for the build tree to hold an importable
# package. It is `ALL` on purpose: an `add_custom_command( OUTPUT ... )` runs
# only when something reachable from the default target depends on its
# output, and kwiver's `python` target is not one -- it is there for
# `make python`. Hanging the copies off that alone left four `.py` files in
# a build tree that needed six hundred, with no error from either the build
# or the configure, because a custom command nobody asks for is not a
# failure, it is just nothing.
if( NOT TARGET viame_python_files )
  add_custom_target( viame_python_files ALL )
endif()

if( NOT TARGET python )
  add_custom_target( python )
endif()

define_property( GLOBAL PROPERTY viame_python_modules
  BRIEF_DOCS "Python extension modules built by VIAME"
  FULL_DOCS  "List of pybind11 extension modules built by VIAME"
  )

source_group( "Python Files" REGULAR_EXPRESSION ".*\\.py$" )

#+
# A filesystem path as a CMake target name: "a/b" becomes "a.b".
#-
macro( _viame_safe_modpath modpath result )
  string( REPLACE "/" "." "${result}" "${modpath}" )
endmacro()

#+
# The python package a module belongs to.
#
# `CMAKE_PROJECT_NAME` is the top-level project, which is VIAME. Kwiver's own
# bindings set `kwiver_python_package` in their directory scope so that they
# stay in the `kwiver` package rather than following the top-level name into
# `viame`. Two copies of one extension module in an interpreter is a pybind11
# duplicate-registration abort, not a warning, which is how that was found.
#-
macro( _viame_python_package result )
  if( kwiver_python_package )
    set( ${result} "${kwiver_python_package}" )
  else()
    string( TOLOWER "${CMAKE_PROJECT_NAME}" ${result} )
  endif()
endmacro()

#+
# Put one python source where the interpreter will find it.
#
#   viame_add_python_module( path modpath module )
#
#   path    the source file
#   modpath the package path below the top package, e.g. `vital/algo`
#   module  the importable name, e.g. `image_object_detector`
#-
function( viame_add_python_module path modpath module )
  _viame_safe_modpath( "${modpath}" safe_modpath )
  _viame_python_package( package )

  set( built
    "${kwiver_python_output_path}/${python_sitename}/${package}/${modpath}/${module}.py" )
  set( install_path
    "${kwiver_python_install_path}/${package}/${modpath}" )

  get_filename_component( built_dir "${built}" DIRECTORY )
  set( name "python-${safe_modpath}-${module}" )

  if( VIAME_PYTHON_SYMLINK )
    if( EXISTS "${built}" AND NOT IS_SYMLINK "${built}" )
      file( REMOVE "${built}" )
    endif()
    add_custom_command(
      OUTPUT  "${built}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${built_dir}"
      COMMAND "${CMAKE_COMMAND}" -E create_symlink "${path}" "${built}"
      DEPENDS "${path}"
      COMMENT "Linking python module ${modpath}/${module}"
      )
  else()
    add_custom_command(
      OUTPUT  "${built}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${built_dir}"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${path}" "${built}"
      DEPENDS "${path}"
      COMMENT "Copying python module ${modpath}/${module}"
      )
  endif()

  add_custom_target( configure-${name} DEPENDS "${built}" SOURCES "${path}" )
  source_group( "Configured Files" FILES "${path}" )

  add_dependencies( viame_python_files configure-${name} )
  add_dependencies( python configure-${name} )

  install( FILES "${built}"
    DESTINATION "${install_path}"
    COMPONENT   runtime
    )
endfunction()

#+
# Build a pybind11 extension module.
#
#   viame_add_python_library( name modpath
#                             [SOURCES ...] [PUBLIC ...] [PRIVATE ...] )
#
# Not `viame_add_library`: an extension module is a MODULE with no version,
# no export, no export header, an output name with no `lib` prefix and a
# platform-specific suffix, landing under site-packages rather than `lib/`.
# Routing it through the general helper meant setting five directory-scope
# variables to turn all of that off.
#-
function( viame_add_python_library name modpath )
  set( multiValueArgs SOURCES PUBLIC PRIVATE )
  cmake_parse_arguments( PYLIB "" "" "${multiValueArgs}" ${ARGN} )

  _viame_safe_modpath( "${modpath}" safe_modpath )
  _viame_python_package( package )

  set( target "python-${safe_modpath}-${name}" )

  add_library( ${target} MODULE ${PYLIB_SOURCES} )

  # pybind11::module rather than Python_LIBRARIES directly: it is what lets
  # the same source build inside a wheel and outside one.
  list( INSERT PYLIB_PRIVATE 0 pybind11::module )
  target_link_libraries( ${target}
    PUBLIC   ${PYLIB_PUBLIC}
    PRIVATE  ${PYLIB_PRIVATE}
    )

  if( MSVC )
    # MSVC cannot compile some of these bindings without the optimizer
    # expanding inline functions, which a debug build otherwise does not do.
    target_compile_options( ${target} PUBLIC "/Ob2" )
  endif()

  if( WIN32 AND NOT CYGWIN )
    set( pysuffix .pyd )
  else()
    set( pysuffix "${CMAKE_SHARED_MODULE_SUFFIX}" )
  endif()

  set( built_dir
    "${kwiver_python_output_path}/${python_sitename}/${package}/${modpath}" )

  # How far `lib/` is from where this module is installed. A module lands at
  #
  #     <prefix>/lib/<python>/site-packages/<package>/<modpath>
  #
  # so the climb is three, plus the package, plus however deep `modpath` goes
  # -- `vital` is one level, `sprokit/pipeline` is two. Counted from the
  # relative layout rather than by subtracting two absolute paths, because
  # the install prefix and `kwiver_python_install_path` are separate cache
  # variables that only happen to agree.
  string( REPLACE "/" ";" modpath_parts "${modpath}" )
  list( LENGTH modpath_parts modpath_depth )
  math( EXPR climb "3 + 1 + ${modpath_depth}" )

  set( to_libs "" )
  foreach( _ RANGE 1 ${climb} )
    string( APPEND to_libs "../" )
  endforeach()
  string( APPEND to_libs "lib" )

  set_target_properties( ${target} PROPERTIES
    OUTPUT_NAME              "${name}"
    PREFIX                   ""
    SUFFIX                   "${pysuffix}"
    LIBRARY_OUTPUT_DIRECTORY "${built_dir}"
    INSTALL_RPATH            "\$ORIGIN/${to_libs}:\$ORIGIN/"
    )

  foreach( config IN LISTS CMAKE_CONFIGURATION_TYPES )
    string( TOUPPER "${config}" upper_config )
    set_target_properties( ${target} PROPERTIES
      "LIBRARY_OUTPUT_DIRECTORY_${upper_config}" "${built_dir}"
      )
  endforeach()

  install( TARGETS ${target}
    LIBRARY DESTINATION "${kwiver_python_install_path}/${package}/${modpath}"
    COMPONENT           runtime
    )

  add_dependencies( python ${target} )
  set_property( GLOBAL APPEND PROPERTY viame_python_modules ${name} )
endfunction()

#+
# Write a package's `__init__.py`, importing the named modules.
#
#   viame_create_python_init( modpath [module ...] )
#
# Written at configure time into the build tree and installed from there. An
# existing file is appended to rather than replaced, and each import is added
# only if it is not already present, so that two calls for one package -- the
# `viame` package is assembled from several directories -- accumulate.
#-
function( viame_create_python_init modpath )
  _viame_python_package( package )

  set( init
    "${kwiver_python_output_path}/${python_sitename}/${package}/${modpath}/__init__.py" )

  if( NOT EXISTS "${init}" )
    if( NOT copyright_header )
      set( copyright_header "# Generated by VIAME" )
    endif()
    file( WRITE "${init}" "${copyright_header}\n\n" )
  endif()

  if( ARGC GREATER 1 )
    file( READ "${init}" contents )

    set( absolute "from __future__ import absolute_import" )
    string( FIND "${contents}" "${absolute}" found )
    if( found EQUAL -1 )
      file( APPEND "${init}" "${absolute}\n\n" )
    endif()

    foreach( module IN LISTS ARGN )
      set( line "from .${module} import *" )
      string( FIND "${contents}" "${line}" found )
      if( found EQUAL -1 )
        file( APPEND "${init}" "${line}\n" )
      endif()
    endforeach()
  endif()

  install( FILES "${init}"
    DESTINATION "${kwiver_python_install_path}/${package}/${modpath}"
    COMPONENT   runtime
    )
endfunction()

#+
# A custom target that runs a command as part of `all`.
#-
function( viame_python_add_command name command comment )
  add_custom_target( ${name} ALL
    COMMAND ${command}
    COMMENT ${comment}
    )
  if( ARGC GREATER 3 )
    add_dependencies( ${name} ${ARGN} )
  endif()
endfunction()
