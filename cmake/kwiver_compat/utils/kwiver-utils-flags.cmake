#
# Function to check validity of compiler flags. If flag is accepted,
# then it is added to the global property.
#

include(CheckCXXCompilerFlag)

define_property(GLOBAL PROPERTY kwiver_warnings
  BRIEF_DOCS "Warning flags for KWIVER build"
  FULL_DOCS "List of warning flags KWIVER will build with."
  )

# Helper function for adding compiler flags
# if a list of flags is supplied, the first valid flag is added
# This is useful if you are looking for the highest level of compiler support
# (-std=c++11  -std=c++0x) will set the flag for the highest level of support.
function(kwiver_check_compiler_flag )
  foreach( flag ${ARGN} )
    string(REPLACE "+" "plus" safeflag "${flag}")
    string(REPLACE "/" "slash" safeflag "${safeflag}")
    check_cxx_compiler_flag("${flag}" "has_compiler_flag-${safeflag}")
    if ( has_compiler_flag-${safeflag} )
      set_property(GLOBAL APPEND PROPERTY kwiver_warnings "${flag}")
      return()
    endif()
  endforeach()
endfunction ()
