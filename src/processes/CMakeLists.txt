project(vistk_processes)

add_subdirectory(helpers)
add_subdirectory(clusters)
add_subdirectory(flow)
add_subdirectory(image)
add_subdirectory(scoring)
add_subdirectory(utilities)

option(VISTK_ENABLE_GRAVL "Enable gravl reader process" OFF)
if (VISTK_ENABLE_GRAVL)
  add_subdirectory(gravl)
endif ()

if (VISTK_ENABLE_TESTING)
  add_subdirectory(examples)
endif ()

set(DOXY_EXCLUDE_PATTERNS
  "*/examples/* */helpers/*")

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  processes
  pipeline)
