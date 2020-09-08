#
# Encapsulation of flags that need to be set for KWIVER under different
# circumstances.
#

include( utils/kwiver-utils-flags )

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  include( kwiver-flags-msvc )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include( kwiver-flags-clang )
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  include( kwiver-flags-gnu )
endif()


get_property( kwiver_cxx_flags GLOBAL PROPERTY kwiver_warnings )
string( REPLACE ";" " " kwiver_cxx_flags "${kwiver_cxx_flags}" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${kwiver_cxx_flags}" )
set( KWIVER_CXX_FLAGS "${kwiver_cxx_flags}" ) # a copy of our custom flags
