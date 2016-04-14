#
# Find PROJ4 library components
#
# Sets:
#   PROJ4_FOUND       to 0 or 1 depending on the result
#   PROJ4_INCLUDE_DIR to directories required for using libproj4
#   PROJ4_LIBRARY     to libproj4 and any dependent library
#
# If PROJ4_REQUIRED is defined, then a fatal error message will be generated if libproj4 is not found
#

if ( NOT PROJ4_INCLUDE_DIR OR NOT PROJ4_LIBRARY OR NOT PROJ4_FOUND )

  # uses externally supplied path. Specifically CMAKE_PREFIX_PATH
  find_library( PROJ4_LIBRARY
    NAMES proj4 libproj4 proj libproj
    DOC "Proj4 specific library file" )

  find_path( PROJ4_INCLUDE_DIR
    NAMES proj4_config.h proj_api.h
    DOC "Path to proj4 include directory" )

  if ( NOT PROJ4_INCLUDE_DIR OR NOT PROJ4_LIBRARY )
    if ( PROJ4_REQUIRED )
      message( FATAL_ERROR "PROJ4 is required and not found." )
    endif ()
  else ()
    set( PROJ4_FOUND 1 )
    mark_as_advanced( PROJ4_FOUND )
  endif ()

endif()

mark_as_advanced( FORCE PROJ4_INCLUDE_DIR )
mark_as_advanced( FORCEBPROJ4_LIBRARY )
