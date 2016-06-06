#
#
#
if (KWIVER_ENABLE_C_BINDINGS)
endif()

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory( c )  # requires python
  add_subdirectory( python )
endif()
