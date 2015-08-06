# KWIVER Python interface support

include(CMakeDependentOption)

cmake_dependent_option( VITAL_ENABLE_PYTHON
  "Enable configuration and installation of KWIVER Python bindings based on C interface"
  ON                  # Default value
  VITAL_ENABLE_C_LIB  # Dependency expression
  OFF                 # State when dependency expression is false
  )

cmake_dependent_option( VITAL_PYTHON_SEARCH_BUILD_DIR
  "Include build directory in search paths when python looks for C interface library."
  ON
  "VITAL_ENABLE_PYTHON;BUILD_SHARED_LIBS OR VITAL_C_LIB_FORCE_SHARED"
  OFF
  )


if( VITAL_ENABLE_PYTHON )
  if( BUILD_SHARED_LIBS    OR     VITAL_C_LIB_FORCE_SHARED )
    if( VITAL_PYTHON_SEARCH_BUILD_DIR )
      set(VITAL_BUILD_DIRECTORY "${VITAL_BINARY_DIR}")
    endif()

    kwiver_configure_file(python-find_vital_libraray.py
      "${CMAKE_CURRENT_SOURCE_DIR}/vital/util/find_vital_library.py.in"
      "${CMAKE_CURRENT_SOURCE_DIR}/vital/util/find_vital_library.py"
      VITAL_BUILD_DIRECTORY
      )

    #TODO: Install rules for python module and files
  else()
    message(WARNING
      "Halting Python bindings processing due to not building shared "
      "libraries. Python cannot load static libraries."
      )
  endif()
endif( VITAL_ENABLE_PYTHON )
