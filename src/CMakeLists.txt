include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_subdirectory(fixes)

add_subdirectory(sprokit)

option(SPROKIT_ENABLE_TOOLS "Build tools" ON)
if (SPROKIT_ENABLE_TOOLS)
  add_subdirectory(tools)
endif ()

add_subdirectory(bindings)
add_subdirectory(processes)
add_subdirectory(schedulers)

# ----------------------------------------------------------------
# build master documentation

sprokit_create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  sprokit-all)
