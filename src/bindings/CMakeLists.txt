project(vistk_bindings)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

if (VISTK_ENABLE_PYTHON)
  add_subdirectory(python)
endif ()
