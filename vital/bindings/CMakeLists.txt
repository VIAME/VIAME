#
# Bindings for VITAL
#

if (KWIVER_ENABLE_C_LIB )
  add_subdirectory( c )
endif()

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif ()
