#
# ARROWS CMake utilities entry point
#

# Pin the current directory as the root directory for ARROWS utility files
set(ARROWS_UTIL_ROOT "${CMAKE_CURRENT_LIST_DIR}")

include("${CMAKE_CURRENT_LIST_DIR}/utils/arrows-utils-buildinfo.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/arrows-utils-targets.cmake")
