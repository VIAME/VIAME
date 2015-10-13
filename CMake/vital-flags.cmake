#
# Encapsulation of flags that need to be set for KWIVER under different
# circumstances.
#

include( utils/kwiver-utils-flags )

if (MSVC)
  include( vital-flags-msvc )
elseif (APPLE)
  include( vital-flags-clang )
else()
  include( vital-flags-gnu )
endif()

get_property( kwiver_cxx_flags GLOBAL PROPERTY kwiver_warnings )
string( REPLACE ";" " " kwiver_cxx_flags "${kwiver_cxx_flags}" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${kwiver_cxx_flags}" )
set( VITAL_CXX_FLAGS ${kwiver_cxx_flags} ) # a copy of our custom flags
