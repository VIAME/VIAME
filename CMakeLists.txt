project(vistk)

cmake_minimum_required(VERSION 2.8)

set(vistk_version_major 0)
set(vistk_version_minor 1)
set(vistk_version_patch 0)
set(vistk_version "${vistk_version_major}.${vistk_version_minor}.${vistk_version_patch}")

# Add the cmake directory for CMake modules.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/cmake/modules"
  ${CMAKE_MODULE_PATH})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(EXECUTABLE_OUTPUT_PATH    ${CMAKE_BINARY_DIR}/bin)
if (WIN32)
  set(LIBRARY_OUTPUT_PATH       ${CMAKE_BINARY_DIR}/bin)
else (WIN32)
  set(LIBRARY_OUTPUT_PATH       ${CMAKE_BINARY_DIR}/lib)
endif (WIN32)
set(MODULE_OUTPUT_PATH        ${LIBRARY_OUTPUT_PATH}/vistk)
set(LUA_OUTPUT_PATH           ${LIBRARY_OUTPUT_PATH}/lua)
set(PYTHON_OUTPUT_PATH        ${LIBRARY_OUTPUT_PATH}/python)
set(DOCUMENTATION_OUTPUT_PATH ${CMAKE_BINARY_DIR}/doc)

set(LUA_VERSION "5.1"
  CACHE STRING "The version of lua to use for bindings")
set(PYTHON_VERSION "2.7"
  CACHE STRING "The version of python to use for bindings")

if (WIN32)
  set(EXECUTABLE_INSTALL_PATH    ${CMAKE_INSTALL_PREFIX}/${PREFIX}/bin)
  set(LIBRARY_INSTALL_PATH       ${CMAKE_INSTALL_PREFIX}/${PREFIX}/bin)
  set(MODULE_INSTALL_PATH        ${CMAKE_INSTALL_PREFIX}/${PREFIX}/bin/${CMAKE_PROJECT_NAME})
  set(LUA_INSTALL_PATH           ${CMAKE_INSTALL_PREFIX}/${PREFIX}/bin/lua${LUA_VERSION})
  set(PYTHON_INSTALL_PATH        ${CMAKE_INSTALL_PREFIX}/${PREFIX}/bin/python${PYTHON_VERSION})
else (WIN32)
  set(EXECUTABLE_INSTALL_PATH    ${CMAKE_INSTALL_PREFIX}/${PREFIX}/bin)
  set(LIBRARY_INSTALL_PATH       ${CMAKE_INSTALL_PREFIX}/${PREFIX}/lib${LIB_SUFFIX})
  set(MODULE_INSTALL_PATH        ${CMAKE_INSTALL_PREFIX}/${PREFIX}/lib${LIB_SUFFIX}/${CMAKE_PROJECT_NAME})
  set(LUA_INSTALL_PATH           ${CMAKE_INSTALL_PREFIX}/${PREFIX}/lib${LIB_SUFFIX}/lua/${LUA_VERSION})
  set(PYTHON_INSTALL_PATH        ${CMAKE_INSTALL_PREFIX}/${PREFIX}/lib${LIB_SUFFIX}/python${PYTHON_VERSION})
endif (WIN32)
set(INCLUDE_INSTALL_PATH       ${CMAKE_INSTALL_PREFIX}/${PREFIX}/include)
set(PIPE_INSTALL_PATH          ${CMAKE_INSTALL_PREFIX}/${PREFIX}/share/${CMAKE_PROJECT_NAME})
set(PIPE_INCLUDE_INSTALL_PATH  ${CMAKE_INSTALL_PREFIX}/${PREFIX}/share/${CMAKE_PROJECT_NAME}/include)
set(DOCUMENTATION_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/${PREFIX}/share/doc/${CMAKE_PROJECT_NAME}-${vistk_version})

set(vistk_warnings
  "")

if (WIN32)
  option(VISTK_ENABLE_DLLHELL_WARNINGS "Enables warnings about DLL visibility" OFF)
  if (NOT VISTK_ENABLE_DLLHELL_WARNINGS)
    # C4251: STL interface warnings
    # C4275: Inheritance warnings
    set(vistk_warnings
      "${vistk_warnings} /wd4251 /wd4275")
  endif (NOT VISTK_ENABLE_DLLHELL_WARNINGS)

  # -----------------------------------------------------------------------------
  # Disable deprecation warnings for standard C and STL functions in VS2005 and
  # later
  # -----------------------------------------------------------------------------
  if (MSVC_VERSION EQUAL 1400 OR MSVC_VERSION GREATER 1400)
    add_definitions(-D_CRT_NONSTDC_NO_DEPRECATE)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SCL_SECURE_NO_DEPRECATE)
  endif (MSVC_VERSION EQUAL 1400 OR MSVC_VERSION GREATER 1400)
else (WIN32)
  set(vistk_warnings
    "${vistk_warnings} -Wextra -Wall")
  set(vistk_warnings
    "${vistk_warnings} -Wctor-dtor-privacy -Woverloaded-virtual -Wformat=2 -Wshadow -Winit-self -Wswitch-enum -Wstrict-overflow=5 -Wcast-align -Wcast-qual -Wfloat-equal -Wabi -Wswitch-default")

  option(VISTK_ENABLE_NITPICK "Generate warnings about nitpicky things" OFF)
  if (VISTK_ENABLE_NITPICK)
    set(vistk_warnings "-Wunsafe-loop-optimizations -Wsign-promo -Winline -Weffc++ ${vistk_warnings}")
  endif (VISTK_ENABLE_NITPICK)

  option(VISTK_ENABLE_PEDANTIC "Be pedantic" OFF)
  if (VISTK_ENABLE_PEDANTIC)
    option(VISTK_ENABLE_PEDANTIC_ERRORS "Be REALLY pedantic" OFF)
    if (VISTK_ENABLE_PEDANTIC_ERRORS)
      set(vistk_warnings "-pedantic-errors ${vistk_warnings}")
    else (VISTK_ENABLE_PEDANTIC_ERRORS)
      set(vistk_warnings "-pedantic ${vistk_warnings}")
    endif (VISTK_ENABLE_PEDANTIC_ERRORS)
  endif (VISTK_ENABLE_PEDANTIC)

  option(VISTK_ENABLE_WERROR "Treat all warnings as errors" OFF)
  if (VISTK_ENABLE_WERROR)
    set(vistk_warnings "-Werror ${vistk_warnings}")
  endif (VISTK_ENABLE_WERROR)

  option(VISTK_ENABLE_NITPICK "Generate warnings about nitpicky things" OFF)
  if (VISTK_ENABLE_NITPICK)
    set(vistk_warnings "-Wunsafe-loop-optimizations -Wsign-promo -Winline -Weffc++ ${vistk_warnings}")
  endif (VISTK_ENABLE_NITPICK)

  option(VISTK_ENABLE_WERROR "Treat all warnings as errors" OFF)
  if (VISTK_ENABLE_WERROR)
    set(vistk_warnings "-Werror ${vistk_warnings}")
  endif (VISTK_ENABLE_WERROR)

  if (CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")
    option(VISTK_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR "Use clang to flag undefined behavior")
    if (VISTK_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR)
      set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} -fcatch-undefined-behavior")
    endif (VISTK_ENABLE_CLANG_CATCH_UNDEFINED_BEHAVIOR)

    option(VISTK_ENABLE_CLANG_ANALYZER "Use clang's static analysis tool")
    if (VISTK_ENABLE_CLANG_ANALYZER)
      set(CMAKE_CXX_FLAGS_DEBUG
        "${CMAKE_CXX_FLAGS_DEBUG} --analyze")
    endif (VISTK_ENABLE_CLANG_ANALYZER)
  elseif (CMAKE_CXX_COMPILER MATCHES "g\\+\\+")
    # GCC-specific warnings
    set(vistk_warnings
      "${vistk_warnings} -std=c++03 -Wlogical-op")
  endif (CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")

  option(VISTK_ENABLE_COVERAGE "Build with coverage testing" OFF)
  if (VISTK_ENABLE_COVERAGE)
    set(vistk_coverage
      "-O0 -pg -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${vistk_coverage}")
  endif (VISTK_ENABLE_COVERAGE)
endif (WIN32)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${vistk_warnings}")

foreach (config ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER "${config}" upper_config)

  set(config_defines
    "VISTK_CONFIGURATION=\"${config}\"")

  if (WIN32)
    set(config_defines
      ${config_defines}
      "VISTK_CONFIGURATION_L=L\"${config}\"")
  endif (WIN32)

  set_directory_properties(
    PROPERTIES
      COMPILE_DEFINITIONS_${upper_config} "${config_defines}")
endforeach (config)

option(VISTK_ENABLE_LUA "Enable Lua bindings" OFF)
option(VISTK_ENABLE_PYTHON "Enable Python bindings" OFF)

option(VISTK_ENABLE_EXTRAS "Enable extra files to help with tooling" OFF)
if (VISTK_ENABLE_EXTRAS)
  add_subdirectory(extra)
endif (VISTK_ENABLE_EXTRAS)

option(VISTK_ENABLE_DOCUMENTATION "Build documentation" OFF)
if (VISTK_ENABLE_DOCUMENTATION)
  include(${CMAKE_SOURCE_DIR}/cmake/doxygen.cmake)
endif (VISTK_ENABLE_DOCUMENTATION)

option(VISTK_ENABLE_TOOLS "Build tools" ON)

set(vistk_boost_version 1.44)

if (VISTK_ENABLE_PYTHON)
  set(vistk_boost_version 1.48)
endif (VISTK_ENABLE_PYTHON)

set(BOOST_ROOT "" CACHE PATH "The root path to Boost")
find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    date_time
    filesystem
    system
    thread)

find_package(VXL REQUIRED)
include(${VXL_CMAKE_DIR}/UseVXL.cmake)

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

set(boost_filesystem_options
  "BOOST_FILESYSTEM_NO_DEPRECATED"
  "BOOST_FILESYSTEM_VERSION=3")

include_directories(SYSTEM ${VXL_CORE_INCLUDE_DIR})
include_directories(SYSTEM ${VXL_VCL_INCLUDE_DIR})
link_directories(${VXL_LIBRARY_DIR})

include(${CMAKE_SOURCE_DIR}/cmake/vxl_tests.cmake)

add_subdirectory(src)

option(VISTK_ENABLE_TESTING "Build tests" OFF)
if (VISTK_ENABLE_TESTING)
  include(${CMAKE_SOURCE_DIR}/cmake/test.cmake)

  # Configure the warning and code coverage suppression file
  configure_file(
    "${CMAKE_SOURCE_DIR}/CTestCustom.cmake.in"
    "${CMAKE_BINARY_DIR}/CTestCustom.cmake"
    COPYONLY)

  add_subdirectory(tests)
endif (VISTK_ENABLE_TESTING)

set(vistk_config_file "${CMAKE_BINARY_DIR}/vistk-config.cmake")
configure_file(
  "${CMAKE_SOURCE_DIR}/vistk-config.cmake.in"
  "${vistk_config_file}")
set(vistk_config_install_file "${CMAKE_BINARY_DIR}/vistk-config-install.cmake")
configure_file(
  "${CMAKE_SOURCE_DIR}/vistk-config-install.cmake.in"
  "${vistk_config_install_file}"
  @ONLY)

# Configure the CMake EXPORT file during installation
install(
  FILES       ${vistk_config_install_file}
  DESTINATION "${LIBRARY_INSTALL_PATH}/cmake"
  COMPONENT   runtime)
install(
  EXPORT      vistk_exports
  DESTINATION "${LIBRARY_INSTALL_PATH}/cmake"
  FILE        vistk-config-targets.cmake
  COMPONENT   runtime)
