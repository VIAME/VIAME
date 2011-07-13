project(vistk_pipeline)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)
