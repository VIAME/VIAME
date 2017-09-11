#
# KWIVER CMake utilities entry point
#

# save this directory so we can find config helper

set( KWIVER_CMAKE_ROOT ${CMAKE_CURRENT_LIST_DIR})

include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-configuration.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-targets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-flags.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-doxygen.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-modules.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-python.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-python-tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/algorithm-utils-targets.cmake")
