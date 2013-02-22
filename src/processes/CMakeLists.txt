project(vistk_processes)

add_subdirectory(helpers)
add_subdirectory(clusters)
add_subdirectory(flow)

if (VISTK_ENABLE_TESTING)
  add_subdirectory(examples)
endif ()

set(DOXY_EXCLUDE_PATTERNS
  "*/examples/* */helpers/*")

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  processes
  pipeline)
