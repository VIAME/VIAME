project(vistk_edges)

add_subdirectory(examples)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    edges
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)
