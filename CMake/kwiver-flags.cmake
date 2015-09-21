#
# Encapsulation of flags that need to be set for KWIVER under different
# circumstances.
#

include(CheckCXXCompilerFlag)

if (MSVC)
  include( kwiver-flags-msvc )
else()
  include( kwiver-flags-gnu )
endif()

get_property(kwiver_cxx_flags GLOBAL PROPERTY kwiver_warnings)
string(REPLACE ";" " " kwiver_cxx_flags "${kwiver_cxx_flags}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${kwiver_cxx_flags}")
