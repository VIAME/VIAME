# Find dependencies.

########################################
# Boost
########################################

set(vistk_boost_version 1.47)

if (VISTK_ENABLE_PYTHON)
  set(vistk_boost_version 1.48)
endif ()

# Required for Boost.Thread.
find_package(Threads REQUIRED)

set(BOOST_ROOT "" CACHE PATH "The root path to Boost")
option(Boost_USE_STATIC_LIBS "Use a statically-linked Boost" OFF)
find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    chrono
    date_time
    filesystem
    system
    thread)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if (WIN32)
  add_definitions(-DBOOST_ALL_NO_LIB)
endif ()

include("${vistk_source_dir}/cmake/snippets/boost_tests.cmake")

########################################
# VXL
########################################

find_package(VXL REQUIRED)
include(${VXL_CMAKE_DIR}/UseVXL.cmake)

include_directories(SYSTEM ${VXL_CORE_INCLUDE_DIR})
include_directories(SYSTEM ${VXL_VCL_INCLUDE_DIR})
link_directories(${VXL_LIBRARY_DIR})

include("${vistk_source_dir}/cmake/snippets/vxl_tests.cmake")
