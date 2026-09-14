# pybind11, vendored, header only
#
# `find_package( pybind11 )` gave two imported targets and this stands in for
# them, because what the packages here actually use is the headers plus the
# interpreter -- no `pybind11_add_module`, no `pybind11_extension`, nothing
# else from its CMake. Checked before writing this: nothing in the tree calls
# one of its functions.
#
#   pybind11::pybind11  headers, and Python's, for an extension module. It
#                       deliberately does **not** link libpython: an
#                       extension is loaded by an interpreter that already
#                       has those symbols, and linking it there breaks on
#                       macOS and is wrong everywhere else.
#   pybind11::embed     the same plus libpython, for a program that starts an
#                       interpreter of its own -- the IQR test is the only
#                       one here.
#
# See README.md for which headers are carried and which are not.

add_library( viame_pybind11 INTERFACE )

target_include_directories( viame_pybind11 SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PYTHON_INCLUDE_DIRS}
  )

add_library( pybind11::pybind11 ALIAS viame_pybind11 )

# What an extension module links. In upstream pybind11 this is the target
# that deliberately leaves libpython out and, on macOS, adds
# `-undefined dynamic_lookup`; on Linux the two are the same thing.
add_library( pybind11::module ALIAS viame_pybind11 )

if( APPLE )
  target_link_options( viame_pybind11 INTERFACE
    "LINKER:-undefined,dynamic_lookup" )
endif()

add_library( viame_pybind11_embed INTERFACE )

target_link_libraries( viame_pybind11_embed INTERFACE
  viame_pybind11
  ${PYTHON_LIBRARIES}
  )

add_library( pybind11::embed ALIAS viame_pybind11_embed )

