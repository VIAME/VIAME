project(vistk_schedulers)

add_subdirectory(examples)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    schedulers
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)
