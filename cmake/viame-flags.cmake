#
# Encapsulation of flags that need to be set for VIAME under different
# circumstances.
#

# Appending to `CMAKE_CXX_FLAGS` and to the linker flags is not idempotent, so
# being included twice would say every flag twice. Harmless to the build and
# confusing to read, and one line stops it. The only include site is
# `viame_project.cmake`, which resets `kwiver_warnings` immediately before --
# so nothing is being guarded out that wanted to run.
include_guard( GLOBAL )

include( viame-flags-check )

set_property( GLOBAL PROPERTY viame_linker_flags )

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  include( viame-flags-msvc )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include( viame-flags-clang )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  include( viame-flags-gnu )
endif()

get_property( viame_cxx_flags GLOBAL PROPERTY kwiver_warnings )
string( REPLACE ";" " " viame_cxx_flags "${viame_cxx_flags}" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${viame_cxx_flags}" )
set( VIAME_CXX_FLAGS ${viame_cxx_flags} ) # a copy of our custom flags

get_property( viame_link_flags GLOBAL PROPERTY viame_linker_flags )
string( REPLACE ";" " " viame_link_flags "${viame_link_flags}" )

foreach( kind EXE SHARED MODULE )
  set( CMAKE_${kind}_LINKER_FLAGS
    "${CMAKE_${kind}_LINKER_FLAGS} ${viame_link_flags}" )
endforeach()

set( VIAME_LINKER_FLAGS ${viame_link_flags} ) # a copy, as above
