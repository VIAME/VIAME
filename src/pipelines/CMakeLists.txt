project(vistk_pipelines)

add_subdirectory(examples)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    pipelines
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)
