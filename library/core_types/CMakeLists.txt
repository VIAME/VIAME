# The types VIAME passes between algorithms
#
# Copied from kwiver's `vital/types` in P5-T02, unchanged: the namespace is
# still `kwiver::vital` and the include guards are still kwiver's, which is
# what lets a translation unit reach both these headers and the ones kwiver's
# own sprokit and arrows headers name without seeing two of anything. Phases 6
# to 8 rewrite the code; this only moves it.
#
# These compile into `viame_algorithm_framework` rather than into a library of
# their own. The two directories name each other -- the types need
# configuration, logging and errors, and the algorithm interfaces are written
# in terms of the types -- and CMake allows a cycle only between static
# libraries. Kwiver cut the same code into eight libraries to keep the graph
# acyclic; which cuts are worth keeping is phase 8's question, and until then
# one library is the honest answer rather than a layering invented here.
#
# The file list is a glob because the set is fixed by
# `design/lite-kwiver-files.txt` and the import script, not chosen here.

file( GLOB core_types_headers CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.txx"
  )

file( GLOB core_types_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
  )

set( VIAME_CORE_TYPES_HEADERS "${core_types_headers}" PARENT_SCOPE )
set( VIAME_CORE_TYPES_SOURCES "${core_types_sources}" PARENT_SCOPE )

kwiver_install_headers(
  SUBDIR     viame/core_types
  ${core_types_headers}
  )
