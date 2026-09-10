project(sprokit_processes)

add_subdirectory(helpers)
add_subdirectory(clusters)
add_subdirectory(examples)

set(DOXY_EXCLUDE_PATTERNS
  "*/examples/* */helpers/*")
