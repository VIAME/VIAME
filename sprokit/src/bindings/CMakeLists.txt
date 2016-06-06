project(sprokit_bindings)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

if (KWIVER_ENABLE_PYTHON)
  add_subdirectory(python)
endif ()
