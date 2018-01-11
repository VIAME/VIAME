#
# Bindings for VITAL
#

if (KWIVER_ENABLE_C_BINDINGS )
  add_subdirectory( c )

  if (KWIVER_ENABLE_PYTHON)
    add_subdirectory( python )
    add_subdirectory( intermediate )
  endif ()
endif()
