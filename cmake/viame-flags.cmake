#
# Encapsulation of flags that need to be set for VIAME under different
# circumstances.
#

include( utils/kwiver-utils-flags )

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
