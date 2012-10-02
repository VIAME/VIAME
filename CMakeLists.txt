project(vistk)

cmake_minimum_required(VERSION 2.8)

option(VISTK_USE_BUILD_TREE "Whether the build tree should be used for default paths" ON)
mark_as_advanced(VISTK_USE_BUILD_TREE)

set(vistk_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(vistk_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")

# Add the cmake directory for CMake modules.
set(CMAKE_MODULE_PATH
  "${vistk_source_dir}/cmake/modules"
  ${CMAKE_MODULE_PATH})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

include(CMakeDependentOption)
include("${vistk_source_dir}/cmake/version.cmake")
include("${vistk_source_dir}/cmake/flags.cmake")
include("${vistk_source_dir}/cmake/configuration.cmake")
include("${vistk_source_dir}/cmake/python.cmake")
include("${vistk_source_dir}/cmake/depends.cmake")

find_package(Git)

include("${vistk_source_dir}/cmake/configure.cmake")
if (GIT_FOUND)
  if (IS_DIRECTORY "${vistk_source_dir}/.git")
    set(vistk_is_in_git ON)
  endif ()
endif ()
if (vistk_is_in_git)
  include("${vistk_source_dir}/cmake/dist.cmake")
  include("${vistk_source_dir}/cmake/hooks.cmake")
endif ()
include("${vistk_source_dir}/cmake/doxygen.cmake")
include("${vistk_source_dir}/cmake/targets.cmake")

source_group("Configured Files"
  REGULAR_EXPRESSION ".*\\.in$")
source_group("Template Files"
  REGULAR_EXPRESSION ".*\\.txx$")

add_subdirectory(conf)
add_subdirectory(src)

option(VISTK_ENABLE_EXTRAS "Enable extra files to help with tooling" OFF)
if (VISTK_ENABLE_EXTRAS)
  add_subdirectory(extra)
endif ()

option(VISTK_ENABLE_TESTING "Build tests" OFF)
if (VISTK_ENABLE_TESTING)
  add_subdirectory(tests)
endif ()
