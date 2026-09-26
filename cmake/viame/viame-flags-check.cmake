# Adding a compiler flag only if the compiler has it
#
# Kwiver's `kwiver-utils-flags.cmake`, renamed. The global property keeps
# kwiver's name for now because `viame-flags.cmake` and the top-level
# CMakeLists both read it and phase 11 is what renames the rest.

include_guard( GLOBAL )
include( CheckCXXCompilerFlag )

define_property( GLOBAL PROPERTY kwiver_warnings
  BRIEF_DOCS "Warning flags for the VIAME build"
  FULL_DOCS  "List of warning flags VIAME will build with"
  )

#+
# Add the first of the given flags the compiler accepts.
#
#   viame_check_compiler_flag( flag [fallback ...] )
#
# Several flags means "the best of these": `-std=c++11 -std=c++0x` takes the
# first the compiler knows. One flag means "this, if it exists".
#-
# ----------------------------------------------------------------------------
# Linker flags VIAME adds, kept apart from the compile flags.
define_property( GLOBAL PROPERTY viame_linker_flags
  BRIEF_DOCS "Flags VIAME adds to every link line"
  FULL_DOCS  "Collected by viame_check_linker_flag and applied by "
             "viame-flags.cmake to the executable, shared and module linker "
             "flags. Separate from kwiver_warnings because a linker flag in "
             "CMAKE_CXX_FLAGS is ignored at compile time and only reaches the "
             "linker because CMake happens to put the compile flags on the "
             "link line too." )

# ----------------------------------------------------------------------------
# Take the first of \p ARGN the compiler will link with, and keep it.
function( viame_check_linker_flag )
  foreach( flag ${ARGN} )
    string( REPLACE "+" "plus" safe "${flag}" )
    string( REPLACE "/" "slash" safe "${safe}" )
    string( REPLACE "," "comma" safe "${safe}" )
    # `check_cxx_compiler_flag` compiles *and links* its probe, so it tests a
    # linker flag as readily as a compile one
    check_cxx_compiler_flag( "${flag}" "has_linker_flag-${safe}" )
    if( has_linker_flag-${safe} )
      set_property( GLOBAL APPEND PROPERTY viame_linker_flags "${flag}" )
      return()
    endif()
  endforeach()
endfunction()

function( viame_check_compiler_flag )
  foreach( flag ${ARGN} )
    string( REPLACE "+" "plus" safe "${flag}" )
    string( REPLACE "/" "slash" safe "${safe}" )
    check_cxx_compiler_flag( "${flag}" "has_compiler_flag-${safe}" )
    if( has_compiler_flag-${safe} )
      set_property( GLOBAL APPEND PROPERTY kwiver_warnings "${flag}" )
      return()
    endif()
  endforeach()
endfunction()
