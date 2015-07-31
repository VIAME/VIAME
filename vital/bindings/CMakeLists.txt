#
# Bindings for VITAL
#

add_subdirectory( c )

if (VITAL_ENABLE_PYTHON)
  add_subdirectory( python )
endif (VITAL_ENABLE_PYTHON)
