project(vistk_processes)

add_subdirectory(helpers)
add_subdirectory(examples)
add_subdirectory(flow)
add_subdirectory(image)
add_subdirectory(utilities)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    processes
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)
