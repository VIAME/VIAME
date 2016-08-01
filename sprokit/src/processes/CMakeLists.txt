project(sprokit_processes)

add_subdirectory(helpers)
add_subdirectory(clusters)
add_subdirectory(flow)

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(examples)
endif ()

set(DOXY_EXCLUDE_PATTERNS
  "*/examples/* */helpers/*")
