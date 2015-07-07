#
# Bindings for VITAL
#

add_subdirectory( c )

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( python )
endif (KWIVER_ENABLE_PYTHON)
