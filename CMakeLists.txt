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

if (WIN32)
  set(destdir bin)
else ()
  set(destdir lib)
endif ()

set(module_output_path "${vistk_binary_dir}/${destdir}/vistk")

include("${vistk_source_dir}/cmake/flags.cmake")
include(CMakeDependentOption)

set(vistk_warnings
  "")

if (MSVC)
  option(VISTK_ENABLE_DLLHELL_WARNINGS "Enables warnings about DLL visibility" OFF)
  if (NOT VISTK_ENABLE_DLLHELL_WARNINGS)
    # C4251: STL interface warnings
    vistk_check_compiler_flag(vistk_warnings /wd4251)
    # C4275: Inheritance warnings
    vistk_check_compiler_flag(vistk_warnings /wd4275)
  endif ()

  # -----------------------------------------------------------------------------
  # Disable deprecation warnings for standard C and STL functions in VS2005 and
  # later
  # -----------------------------------------------------------------------------
  if (MSVC_VERSION GREATER 1400 OR
      MSVC_VERSION EQUAL 1400)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
  endif ()
# Assume GCC-compatible
else ()
  set(vistk_using_clang FALSE)

  check_cxx_compiler_flag(-std=c++11 has_cxx11_flags)

  cmake_dependent_option(VISTK_ENABLE_CXX11 "Enable compilation with C++11 support" OFF
    has_cxx11_flags OFF)

  # Check for clang
  if (CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")
    set(vistk_using_clang TRUE)
  else ()
    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}"
              -dumpversion
      WORKING_DIRECTORY
              "${vistk_source_dir}"
      RESULT_VARIABLE
              gcc_return
      OUTPUT_VARIABLE
              gcc_version)

    if (gcc_version VERSION_GREATER "4.7.0" AND
        gcc_version VERSION_LESS "4.7.2")
      if (VISTK_ENABLE_CXX11)
        message(ERROR
          "C++11 ABI is broken with GCC 4.7.0 and 4.7.1."
          "Refusing to enable C++11 support.")
      endif ()
    endif ()
  endif ()

  # Check for GCC-compatible visibility settings
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-fvisibility=hidden VISTK_HAVE_GCC_VISIBILITY)

  # Hide symbols by default
  vistk_check_compiler_flag(vistk_warnings -fvisibility=hidden)
  # Set the standard to C++98
  if (VISTK_ENABLE_CXX11)
    vistk_check_compiler_flag(vistk_warnings -std=c++11)
  else ()
    vistk_check_compiler_flag(vistk_warnings -std=c++98)
  endif ()
  # General warnings
  vistk_check_compiler_flag(vistk_warnings -Wall)
  vistk_check_compiler_flag(vistk_warnings -Wextra)
  # Class warnings
  vistk_check_compiler_flag(vistk_warnings -Wabi)
  vistk_check_compiler_flag(vistk_warnings -Wctor-dtor-privacy)
  vistk_check_compiler_flag(vistk_warnings -Winit-self)
  vistk_check_compiler_flag(vistk_warnings -Woverloaded-virtual)
  # Pointer warnings
  vistk_check_compiler_flag(vistk_warnings -Wpointer-arith)
  vistk_check_compiler_flag(vistk_warnings -Wstrict-null-sentinel)
  # Enumeration warnings
  vistk_check_compiler_flag(vistk_warnings -Wswitch-default)
  vistk_check_compiler_flag(vistk_warnings -Wswitch-enum)
  # Formatting warnings
  vistk_check_compiler_flag(vistk_warnings -Wformat-security)
  vistk_check_compiler_flag(vistk_warnings -Wformat=2)
  # Casting warnings
  vistk_check_compiler_flag(vistk_warnings -Wcast-align)
  vistk_check_compiler_flag(vistk_warnings -Wcast-qual)
  vistk_check_compiler_flag(vistk_warnings -Wdouble-promotion)
  vistk_check_compiler_flag(vistk_warnings -Wfloat-equal)
  vistk_check_compiler_flag(vistk_warnings -Wstrict-overflow=5)
  # TODO: Python triggers warnings with this
  #vistk_check_compiler_flag(vistk_warnings -Wold-style-cast)
  # Variable naming warnings
  vistk_check_compiler_flag(vistk_warnings -Wshadow)
  # C++ 11 compatibility warnings
  vistk_check_compiler_flag(vistk_warnings -Wnarrowing)
  # Exception warnings
  vistk_check_compiler_flag(vistk_warnings -Wnoexcept)
  # Miscellaneous warnings
  vistk_check_compiler_flag(vistk_warnings -Wlogical-op)
  vistk_check_compiler_flag(vistk_warnings -Wmissing-braces)

  option(VISTK_ENABLE_NITPICK "Generate warnings about nitpicky things" OFF)
  if (VISTK_ENABLE_NITPICK)
    vistk_check_compiler_flag(vistk_warnings -Wunsafe-loop-optimizations)
    vistk_check_compiler_flag(vistk_warnings -Wsign-promo)
    vistk_check_compiler_flag(vistk_warnings -Winline)
    vistk_check_compiler_flag(vistk_warnings -Weffc++)
  endif ()

  option(VISTK_ENABLE_PEDANTIC "Be pedantic" OFF)
  cmake_dependent_option(VISTK_ENABLE_PEDANTIC_ERRORS "Be REALLY pedantic" OFF
    VISTK_ENABLE_PEDANTIC OFF)
  if (VISTK_ENABLE_PEDANTIC)
    if (VISTK_ENABLE_PEDANTIC_ERRORS)
      vistk_check_compiler_flag(vistk_warnings -pedantic-errors)
    else ()
      vistk_check_compiler_flag(vistk_warnings -pedantic)
    endif ()
  endif ()

  option(VISTK_ENABLE_WERROR "Treat all warnings as errors" OFF)
  if (VISTK_ENABLE_WERROR)
    vistk_check_compiler_flag(vistk_warnings -Werror)
  endif ()

  cmake_dependent_option(VISTK_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR "Use clang to flag undefined behavior" OFF
    vistk_using_clang OFF)
  if (VISTK_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR)
    vistk_check_compiler_flag(vistk_warnings -fcatch-undefined-behavior)
  endif ()

  option(VISTK_ENABLE_COVERAGE "Build with coverage testing" OFF)
  if (VISTK_ENABLE_COVERAGE)
    set(vistk_coverage
      "")

    vistk_check_compiler_flag(vistk_coverage -O0)
    vistk_check_compiler_flag(vistk_coverage -pg)
    vistk_check_compiler_flag(vistk_coverage -fprofile-arcs)
    vistk_check_compiler_flag(vistk_coverage -ftest-coverage)

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${vistk_coverage}")
  endif ()
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${vistk_warnings}")

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
include(${VXL_CMAKE_DIR}/UseVXL.cmake)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

if (WIN32)
  add_definitions(-DBOOST_ALL_NO_LIB)
endif ()

include_directories(SYSTEM ${VXL_CORE_INCLUDE_DIR})
include_directories(SYSTEM ${VXL_VCL_INCLUDE_DIR})
link_directories(${VXL_LIBRARY_DIR})

find_package(Git)

include("${vistk_source_dir}/cmake/vxl_tests.cmake")

include("${vistk_source_dir}/cmake/configure.cmake")
if (GIT_FOUND)
  if (IS_DIRECTORY "${vistk_source_dir}/.git")
    include("${vistk_source_dir}/cmake/dist.cmake")
  endif ()
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
