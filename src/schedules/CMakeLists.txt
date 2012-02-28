project(vistk_schedules)

add_subdirectory(examples)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    schedules
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)
