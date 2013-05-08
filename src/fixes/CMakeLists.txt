project(sprokit_fixes)

set(sprokit_fix_source_dir "${CMAKE_CURRENT_SOURCE_DIR}"
  CACHE INTERNAL "Source fix include directory path")
set(sprokit_fix_binary_dir "${CMAKE_CURRENT_BINARY_DIR}"
  CACHE INTERNAL "Configured fix include directory path")

if (SPROKIT_ENABLE_PYTHON)
  include("${CMAKE_CURRENT_SOURCE_DIR}/python.cmake")
endif ()
include("${CMAKE_CURRENT_SOURCE_DIR}/tools.cmake")
