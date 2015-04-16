#
# Encapsulation of flags that need to be set for KWIVER under different
# circumstances.
#

include(CheckCXXCompilerFlag)

define_property(GLOBAL PROPERTY kwiver_warnings
  BRIEF_DOCS "Warning flags for KWIVER build"
  FULL_DOCS "List of warning flags KWIVER will build with."
  )

# Helper function for adding compiler flags
function(kwiver_check_compiler_flag flag)
  string(REPLACE "+" "plus" safeflag "${flag}")
  check_cxx_compiler_flag("${flag}" "has_compiler_flag-${safeflag}")
  if ( has_compiler_flag-${safeflag} )
    set_property(GLOBAL APPEND PROPERTY kwiver_warnings "${flag}")
  endif()
endfunction ()

if (MSVC)
  include( kwiver-flags-msvc )
else()
  include( kwiver-flags-gnu )
endif()

get_property(kwiver_cxx_flags GLOBAL PROPERTY kwiver_warnings)
string(REPLACE ";" " " kwiver_cxx_flags "${kwiver_cxx_flags}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${kwiver_cxx_flags}")
