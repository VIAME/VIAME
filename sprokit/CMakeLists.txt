project(sprokit)

cmake_minimum_required(VERSION 3.0)

set(sprokit_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(sprokit_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")

##?++  temp needed until new macros are used
check_cxx_compiler_flag(-fvisibility=hidden SPROKIT_HAVE_GCC_VISIBILITY)


## Copy options from KWIVER
set(SPROKIT_ENABLE_PYTHON ${KWIVER_ENABLE_PYTHON})

include(CMakeDependentOption)

## set(sprokit_output_dir         "${sprokit_binary_dir}")
set(sprokit_output_dir         "${KWIVER_BINARY_DIR}")


## TODO replace sprokit macros with kwiver macros if possible
# Sprokit macros
include("${sprokit_source_dir}/conf/sprokit-macro-configure.cmake")
include("${sprokit_source_dir}/conf/sprokit-macro-targets.cmake")
include("${sprokit_source_dir}/conf/sprokit-macro-python.cmake")

# Support functions.
include("${sprokit_source_dir}/cmake/support/pkgconfig.cmake")
include("${sprokit_source_dir}/cmake/support/doxygen.cmake")

### TODO This will not work without going to root directory
find_package(Git)

if (GIT_FOUND)
  if (IS_DIRECTORY "${sprokit_source_dir}/.git")
    set(sprokit_is_in_git ON)
  endif ()
endif ()

# Project setup.
include(${sprokit_source_dir}/cmake/snippets/configuration.cmake)
include(${sprokit_source_dir}/cmake/snippets/groups.cmake)

if (sprokit_is_in_git)
  include(${sprokit_source_dir}/cmake/snippets/dist.cmake)
endif ()

# Dependencies.
include(${sprokit_source_dir}/cmake/snippets/python.cmake)

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
