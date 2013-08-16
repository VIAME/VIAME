project(sprokit)

cmake_minimum_required(VERSION 2.8.7)

option(SPROKIT_USE_BUILD_TREE "Whether the build tree should be used for default paths" ON)
mark_as_advanced(SPROKIT_USE_BUILD_TREE)

set(sprokit_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(sprokit_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")

# Add the cmake directory for CMake modules.
list(INSERT CMAKE_MODULE_PATH 0
  "${sprokit_source_dir}/cmake/modules")

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

include(CMakeDependentOption)

set(sprokit_output_dir
  "${sprokit_binary_dir}")

# Sprokit macros
include("${sprokit_source_dir}/conf/sprokit-macro-configure.cmake")
include("${sprokit_source_dir}/conf/sprokit-macro-targets.cmake")
include("${sprokit_source_dir}/conf/sprokit-macro-python.cmake")

# Support functions.
include("${sprokit_source_dir}/cmake/support/pkgconfig.cmake")
include("${sprokit_source_dir}/cmake/support/doxygen.cmake")

find_package(Git)

if (GIT_FOUND)
  if (IS_DIRECTORY "${sprokit_source_dir}/.git")
    set(sprokit_is_in_git ON)
  endif ()
endif ()

# Project setup.
include("${sprokit_source_dir}/cmake/snippets/version.cmake")
include("${sprokit_source_dir}/cmake/snippets/flags.cmake")
include("${sprokit_source_dir}/cmake/snippets/configuration.cmake")
include("${sprokit_source_dir}/cmake/snippets/groups.cmake")
if (sprokit_is_in_git)
  include("${sprokit_source_dir}/cmake/snippets/dist.cmake")
  include("${sprokit_source_dir}/cmake/snippets/hooks.cmake")
endif ()

# Dependencies.
include("${sprokit_source_dir}/cmake/snippets/python.cmake")
include("${sprokit_source_dir}/cmake/snippets/depends.cmake")

set(export_name
  sprokit_exports)

add_subdirectory(src)

option(SPROKIT_ENABLE_EXTRAS "Enable extra files to help with tooling" OFF)
if (SPROKIT_ENABLE_EXTRAS)
  add_subdirectory(extra)
endif ()

option(SPROKIT_ENABLE_TESTING "Build tests" OFF)
if (SPROKIT_ENABLE_TESTING)
  include(CTest)

  add_subdirectory(tests)
endif ()

add_subdirectory(conf)
