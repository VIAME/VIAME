project(vistk)

cmake_minimum_required(VERSION 2.8)

option(VISTK_USE_BUILD_TREE "Whether the build tree should be used for default paths" ON)
mark_as_advanced(VISTK_USE_BUILD_TREE)

set(vistk_version_major 0)
set(vistk_version_minor 1)
set(vistk_version_patch 0)
set(vistk_version "${vistk_version_major}.${vistk_version_minor}.${vistk_version_patch}")

set(vistk_source_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(vistk_binary_dir "${CMAKE_CURRENT_BINARY_DIR}")

# Add the cmake directory for CMake modules.
set(CMAKE_MODULE_PATH
  "${vistk_source_dir}/cmake/modules"
  ${CMAKE_MODULE_PATH})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

include(CMakeDependentOption)
include("${vistk_source_dir}/cmake/flags.cmake")

foreach (config ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER "${config}" upper_config)

  set(config_defines
    "VISTK_CONFIGURATION=\"${config}\""
    "VISTK_CONFIGURATION_L=L\"${config}\"")

  set_directory_properties(
    PROPERTIES
      COMPILE_DEFINITIONS_${upper_config} "${config_defines}")
endforeach ()

option(VISTK_ENABLE_PYTHON "Enable Python bindings" OFF)
cmake_dependent_option(VISTK_ENABLE_PYTHON3 "Use Python3" OFF
  VISTK_ENABLE_PYTHON OFF)
if (VISTK_ENABLE_PYTHON)
  set(python_output_path "${vistk_binary_dir}/${destdir}/python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")

  set(PYTHON_VERSION "2.7"
    CACHE STRING "The version of python to use for bindings")
  set(PYTHON_ABIFLAGS ""
    CACHE STRING "The ABI flags for the version of Python being used")

  if (VISTK_ENABLE_PYTHON3)
    set(Python_ADDITIONAL_VERSIONS
      3
      ${PYTHON_VERSION})
  endif ()

  # This is to avoid Boost.Python's headers to have __declspec(dllimport) in
  # the headers which confuses Visual Studio's linker.
  cmake_dependent_option(VISTK_HACK_LINK_BOOST_PYTHON_STATIC "Link Boost.Python statically" ON
    WIN32 OFF)
  mark_as_advanced(VISTK_HACK_LINK_BOOST_PYTHON_STATIC)
  if (VISTK_HACK_LINK_BOOST_PYTHON_STATIC)
    add_definitions(-DBOOST_PYTHON_STATIC_LIB)
  endif ()

  if (CMAKE_VERSION VERSION_LESS "2.8.8")
    message(WARNING "Python 3 support may not work with CMake versions older than 2.8.8")
  endif ()
endif ()

option(VISTK_ENABLE_EXTRAS "Enable extra files to help with tooling" OFF)
if (VISTK_ENABLE_EXTRAS)
  add_subdirectory(extra)
endif ()

option(VISTK_ENABLE_TOOLS "Build tools" ON)

set(vistk_boost_version 1.47)

if (VISTK_ENABLE_PYTHON)
  set(vistk_boost_version 1.48)
endif ()

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

find_package(VXL REQUIRED)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if (WIN32)
  add_definitions(-DBOOST_ALL_NO_LIB)
endif ()

include_directories(SYSTEM ${VXL_CORE_INCLUDE_DIR})
include_directories(SYSTEM ${VXL_VCL_INCLUDE_DIR})
link_directories(${VXL_LIBRARY_DIR})

find_package(Git)

include("${vistk_source_dir}/cmake/boost_tests.cmake")
include("${vistk_source_dir}/cmake/vxl_tests.cmake")

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

option(VISTK_ENABLE_TESTING "Build tests" OFF)
if (VISTK_ENABLE_TESTING)
  include("${vistk_source_dir}/cmake/test.cmake")

  add_subdirectory(tests)
endif ()
