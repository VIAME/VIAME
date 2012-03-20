project(vistk_processes)

add_subdirectory(helpers)
add_subdirectory(flow)
add_subdirectory(image)
add_subdirectory(utilities)

if (VISTK_ENABLE_TESTS)
  add_subdirectory(examples)
endif (VISTK_ENABLE_TESTS)

if (VISTK_ENABLE_DOCUMENTATION)
  set(DOXY_EXCLUDE_PATTERNS
    "*/examples/* */helpers/*")

  create_doxygen(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    processes
    pipeline)
endif (VISTK_ENABLE_DOCUMENTATION)
