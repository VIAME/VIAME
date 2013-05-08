include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

set(sprokit_fix_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(sprokit_fix_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")

if (SPROKIT_ENABLE_PYTHON)
  include("${CMAKE_CURRENT_SOURCE_DIR}/fixes/python.cmake")
endif ()
include("${CMAKE_CURRENT_SOURCE_DIR}/fixes/tools.cmake")

add_subdirectory(sprokit)

option(SPROKIT_ENABLE_TOOLS "Build tools" ON)
if (SPROKIT_ENABLE_TOOLS)
  add_subdirectory(tools)
endif ()

add_subdirectory(bindings)
add_subdirectory(processes)
add_subdirectory(schedulers)
