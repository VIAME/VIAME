###
# `viame.types`
#
# The bindings live beside the C++ they bind since P8-T01, rather than in
# `python/kwiver/vital/types`, which is what the copy from kwiver left. Each
# file keeps its name with a `_python` suffix -- the convention
# `library/file_io/opencv_yaml_python.cxx` already set -- because 47 of them
# would otherwise collide with the C++ source of the very type they bind.
#
# **The module path is unchanged.** `viame_add_python_library` takes it as an
# argument rather than deriving it from the source location, so
# `viame.types.bounding_box` is still `viame.types.bounding_box`
# and the four files in the tree that import a submodule by name keep working.
# `library/core_types/tests/test_python_types.py` holds that surface to what
# it was: 106 classes and 965 members, recorded before this moved.
##

# `${PYTHON_LIBRARIES}` goes on every module below, because VIAME links with
# `-Wl,--no-undefined` and an extension module leaves the interpreter's
# symbols to be resolved at import. In `python/` the flag was stripped from
# the directory's link flags instead; linking libpython is what
# `library/file_io` already does for `_opencv_yaml`, and it does not weaken
# the check for the C++ in this directory the way stripping the flag would.
#
# The package these install into used to be set here, to `kwiver`. Nothing
# sets it now: P11-T02 made the whole tree one package, derived from the
# project name.
set( THIS_MODULE types )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/types_init.py "${THIS_MODULE}" __init__ )

# ----------------------------------------------------------------------------
# One extension module, not fifty-six
#
# Each of these was its own `.so`, and fifty-six of them cost 68.6 MB because
# every module carries its own copy of the same instantiated pybind11 and STL
# templates. Linked together the code is 16.6 MB, 8.3 MB stripped: 83% of the
# exported symbols were duplicates. Measured with the linker on the existing
# object files before the change was written, not estimated.
#
# `viame.types.<name>` is unchanged. Each name is a submodule of `_types` and
# a one-line `viame/types/<name>.py` re-exports it, so the import path, the
# `from viame.types.X import *` lines in `types_init.py`, and the four files
# in the tree that import a submodule by name all keep working.
# `tests/test_python_types.py` holds the surface to 106 classes and 965
# members and is what says whether that is true.
#
# The order comes from `types_init.py` rather than a list kept here, because
# it is load-bearing -- `detected_object` references `detected_object_type`,
# `homography` references `transform_2d` -- and two lists that must agree
# eventually do not.
# ----------------------------------------------------------------------------

set( vital_python_headers
     image_python.h
     image_container_python.h
  )

# The module list, and the order they register in, both come from the sources
# -- `VIAME_PYTHON_MODULE` names each module and `VIAME_PYTHON_REQUIRE` names
# what it needs -- so a binding that grows a dependency is ordered correctly
# by saying so where it already had to. `types_init.py`'s order will not do:
# seventeen of these dependencies are back edges in it. See
# `generate_types_fold.py`.
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/generate_types_fold.py"
          --source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
          --output     "${CMAKE_CURRENT_BINARY_DIR}/types_fold_python.cxx"
  RESULT_VARIABLE viame_types_fold_result
  OUTPUT_VARIABLE viame_types_fold_output
  ERROR_VARIABLE  viame_types_fold_error
  )
if( NOT viame_types_fold_result EQUAL 0 )
  message( FATAL_ERROR
    "generate_types_fold.py failed (${viame_types_fold_result}):\n"
    "${viame_types_fold_output}${viame_types_fold_error}" )
endif()
message( STATUS "${viame_types_fold_output}" )

# Re-run it when a binding is added, removed, or changes its dependencies.
# Configure-time generation is otherwise invisible to the build.
file( GLOB viame_types_binding_sources
      CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*_python.cxx" )
set_property( DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
              ${viame_types_binding_sources}
              "${CMAKE_CURRENT_SOURCE_DIR}/generate_types_fold.py" )

# The submodules the generator emitted, which is also the source list: one
# `<name>_python.cxx` each, `types` being `types_module_python.cxx`. Read back
# from the generated file rather than globbing `*_python.cxx`, because five of
# those are dead -- `geo_MGRS`, `geo_covariance`, `homography_f2w`, `mesh` and
# `descriptor_class` have had no object in the build for as long as the build
# has existed, and two of them no longer compile at all. A glob would have
# quietly started building them.
file( STRINGS "${CMAKE_CURRENT_BINARY_DIR}/types_fold_python.cxx"
      viame_types_submodule_lines
      REGEX "def_submodule\\( \"[A-Za-z_0-9]+\" \\)" )

set( viame_types_modules )
foreach( line IN LISTS viame_types_submodule_lines )
  string( REGEX REPLACE ".*def_submodule\\( \"([A-Za-z_0-9]+)\" \\).*" "\\1"
          name "${line}" )
  list( APPEND viame_types_modules "${name}" )
endforeach()

list( LENGTH viame_types_modules viame_types_count )
if( viame_types_count LESS 50 )
  message( FATAL_ERROR
    "core_types: read ${viame_types_count} submodules from the generated "
    "fold, expected at least 50" )
endif()

# `image` and `image_container` are translation units of the `types` module
# rather than modules of their own, so they are named and not derived.
set( viame_types_sources
     ${vital_python_headers}
     image_python.cxx
     image_container_python.cxx
     "${CMAKE_CURRENT_BINARY_DIR}/types_fold_python.cxx"
  )
foreach( name IN LISTS viame_types_modules )
  if( name STREQUAL "types" )
    list( APPEND viame_types_sources types_module_python.cxx )
  else()
    list( APPEND viame_types_sources "${name}_python.cxx" )
  endif()
endforeach()

viame_add_python_library(
  _types
  "${THIS_MODULE}"
  SOURCES ${viame_types_sources}
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          viame_algorithm_framework
  )

# What turns `VIAME_PYTHON_MODULE` from `PYBIND11_MODULE` into a registration
# function; see `python_fold.h`. Set on the target rather than globally, so
# that a translation unit outside this fold still builds its own module.
target_compile_definitions( python-types-_types PRIVATE VIAME_PYTHON_FOLD )
target_include_directories( python-types-_types PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}" )

# The re-export shims. Generated rather than committed: fifty-six one-line
# files that must agree with the list above are a thing to derive, not to
# maintain.
foreach( name IN LISTS viame_types_modules )
  set( shim "${CMAKE_CURRENT_BINARY_DIR}/shims/${name}.py" )
  file( CONFIGURE OUTPUT "${shim}"
        CONTENT "# Generated by VIAME. viame.types.${name} is a submodule of the
# folded `_types` extension module; see python.cmake and python_fold.h.
from viame.types._types.${name} import *  # noqa: F401,F403
" )
  viame_add_python_module( "${shim}" "${THIS_MODULE}" "${name}" )
endforeach()
