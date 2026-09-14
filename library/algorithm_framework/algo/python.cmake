###
# `kwiver.vital.algo`
#
# The bindings live beside the interfaces they bind since P8-T02, rather than
# in `python/kwiver/vital/algo`, and they are **source** now rather than
# output: castxml and pygccxml parsed every header at configure time and
# wrote these, and that whole machinery is gone.
#
# What was deleted with it, from `python/kwiver/vital/algo`:
# `cpp_to_pybind11.py`, `config.ini.in`, and 130 lines of CMake spent
# persuading castxml's bundled clang 13 to parse a standard library it was
# never going to like -- emulating the host compiler, then emulating none,
# then hunting the C++ include directories by regex. All of that ran on every
# configure to produce files that never changed.
#
# The committed bindings are byte-for-byte what the generator last produced,
# minus its provenance comment and with sibling includes made local. They
# were good code: idiomatic pybind11 with the interface docstrings carried
# over. `library/core_types/tests/test_python_types.py` holds the 44 exported
# classes and their members to what they were.
#
# Each file keeps its name with a `_python` suffix, as the type bindings do,
# because otherwise every one of them collides with the interface header it
# binds.
#
# Adding an interface now means writing its binding and its trampoline and
# naming them in `algorithm_module_python.cxx` -- three places, once, rather
# than a code generator in the build.
##

# `CMAKE_CURRENT_LIST_DIR`, not `CMAKE_CURRENT_SOURCE_DIR`: this file is
# `include()`d from the directory above, and an include does not change the
# current source directory. The difference is a configure error naming a file
# one directory too high.
set( THIS_MODULE vital/algo )

# As in `library/core_types/python.cmake`: the package these install into,
# and libpython, neither of which this directory inherits from `python/`
set( kwiver_python_package "kwiver" )

viame_add_python_module(
  ${CMAKE_CURRENT_LIST_DIR}/algo_init.py
  "${THIS_MODULE}"
  __init__ )

file( GLOB _algo_python_sources
  "${CMAKE_CURRENT_LIST_DIR}/*_python.cxx" )

file( GLOB _algo_python_headers
  "${CMAKE_CURRENT_LIST_DIR}/*_python.h"
  "${CMAKE_CURRENT_LIST_DIR}/*_python.txx" )

viame_add_python_library(
  algos
  "${THIS_MODULE}"
  SOURCES ${_algo_python_sources}
          ${_algo_python_headers}
  PRIVATE pybind11::pybind11
          ${PYTHON_LIBRARIES}
          vital
          vital_config
          vital_algo
)
