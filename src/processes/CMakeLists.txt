project(vistk_processes)

add_subdirectory(examples)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    processes
    pipeline
    pipeline_types)
endif (VISTK_ENABLE_DOCUMENTATION)
