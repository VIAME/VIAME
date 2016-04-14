#
# Bindings for VITAL
#

include(CMakeDependentOption)

cmake_dependent_option( VITAL_ENABLE_PYTHON
  "Enable configuration and installation of KWIVER Python bindings based on C interface"
  ON                  # Default value
  VITAL_ENABLE_C_LIB  # Dependency expression
  OFF                 # State when dependency expression is false
  )

cmake_dependent_option( VITAL_PYTHON_SEARCH_BUILD_DIR
  "Include build directory in search paths when python looks for C interface library."
  ON                            # default value
  VITAL_ENABLE_PYTHON           # dependency
  OFF                           # State when dependency expression is false
  )

option(VITAL_ENABLE_C_LIB
  "Enable building of C interface library."
  OFF
  )


if (VITAL_ENABLE_C_LIB )
  add_subdirectory( c )

  if (VITAL_ENABLE_PYTHON)
    add_subdirectory( python )
  endif ()

else()
  unset( VITAL_ENABLE_PYTHON CACHE )
endif()
