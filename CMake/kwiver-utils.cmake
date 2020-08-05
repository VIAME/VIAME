#
# KWIVER CMake utilities entry point
#

# save this directory so we can find config helper

set( KWIVER_CMAKE_ROOT ${CMAKE_CURRENT_LIST_DIR})

if (KWIVER_ENABLE_PYTHON)
  include( kwiver-setup-python )
  include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-tests-python.cmake")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-configuration.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-targets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-flags.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-doxygen.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-modules.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-sphinx.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-python.cmake")
if(MSVC)
  include("${CMAKE_CURRENT_LIST_DIR}/utils/kwiver-utils-msvc.cmake")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/utils/algorithm-utils-targets.cmake")
