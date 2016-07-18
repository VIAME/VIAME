#
# Bindings for VITAL
#

include(CMakeDependentOption)

option(VITAL_ENABLE_C_LIB
  "Enable building of C interface library."
  OFF
  )

cmake_dependent_option( VITAL_ENABLE_PYTHON
  "Enable configuration and installation of KWIVER Python bindings based on C interface"
  ON                  # Default value
  VITAL_ENABLE_C_LIB  # Dependency expression
  OFF                 # State when dependency expression is false
  )


if (VITAL_ENABLE_C_LIB )
  add_subdirectory( c )

  if (VITAL_ENABLE_PYTHON)
    add_subdirectory( python )
  endif ()

else()
  unset( VITAL_ENABLE_PYTHON CACHE )
endif()
