project(vistk)

cmake_minimum_required(VERSION 2.8.7)

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

# Support functions.
include("${vistk_source_dir}/cmake/support/configure.cmake")
include("${vistk_source_dir}/cmake/support/doxygen.cmake")
include("${vistk_source_dir}/cmake/support/targets.cmake")

find_package(Git)

if (GIT_FOUND)
  if (IS_DIRECTORY "${vistk_source_dir}/.git")
    set(vistk_is_in_git ON)
  endif ()
endif ()

option(VISTK_ENABLE_GRAVL "Enable gravl reader process" OFF)
if (VISTK_ENABLE_GRAVL)
  find_package(gravl REQUIRED)

  # Check if C++11 support is required by gravl, if not using MSVC. (On MSVC,
  # there are no options for it, so nothing we can do without knowing the
  # respective compiler versions; we will be okay as long as we are using the
  # same or never version as was used to build gravl; otherwise gravl's
  # build-time check catch any incompatibility).
  if (NOT MSVC)
    if (GRAVL_REQUIRE_CPP11)
      if (NOT has_cxx11_flags)
        message(SEND_ERROR
          "Found gravl built with C++11 support, but your compiler has no option "
          "to enable C++11 support. This configuration is not supported.")
      elseif (NOT VISTK_ENABLE_CXX11)
        message(SEND_ERROR
          "Found gravl built with C++11 support, but VISTK_ENABLE_CXX11 is FALSE. "
          "This configuration is not supported.")
      endif ()
    endif ()
  endif ()
endif ()

# Project setup.
include("${vistk_source_dir}/cmake/snippets/version.cmake")
include("${vistk_source_dir}/cmake/snippets/flags.cmake")
include("${vistk_source_dir}/cmake/snippets/configuration.cmake")
include("${vistk_source_dir}/cmake/snippets/groups.cmake")
if (vistk_is_in_git)
  include("${vistk_source_dir}/cmake/snippets/dist.cmake")
  include("${vistk_source_dir}/cmake/snippets/hooks.cmake")
endif ()

# Dependencies.
include("${vistk_source_dir}/cmake/snippets/python.cmake")
include("${vistk_source_dir}/cmake/snippets/depends.cmake")

add_subdirectory(conf)
add_subdirectory(src)

option(VISTK_ENABLE_EXTRAS "Enable extra files to help with tooling" OFF)
if (VISTK_ENABLE_EXTRAS)
  add_subdirectory(extra)
endif ()

option(VISTK_ENABLE_TESTING "Build tests" OFF)
if (VISTK_ENABLE_TESTING)
  include(CTest)

  add_subdirectory(tests)
endif ()
