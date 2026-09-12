# Where tests are built and run from
#
# Included by each test directory's CMakeLists. `no_install` is gone with
# kwiver's `kwiver_install` wrapper that read it -- test executables are not
# installed because nothing installs them, not because a variable said so.

if( NOT TARGET GTest::gtest )
  # GoogleTest is built by `third_party/googletest`, added by the top-level
  # CMakeLists when tests are enabled. It was `find_package( GTest REQUIRED )`
  # resolved through fletch's prefix, and was the last thing in VIAME's own
  # build that needed fletch at all.
  message( FATAL_ERROR "third_party/googletest has not been added yet" )
endif()

# `CMAKE_BINARY_DIR`, not `VIAME_BINARY_DIR`: the latter is
# `${CMAKE_BINARY_DIR}/build`, where the superbuild stages things, and using
# it here would move every test binary one directory down from where ctest
# has always looked for it.
if( WIN32 )
  # TODO: output to a per-configuration directory and use $<CONFIG> in the
  # working path, once generator expressions work in test properties.
  set( viame_test_output_path "${CMAKE_BINARY_DIR}/bin" )
else()
  set( viame_test_output_path  "${CMAKE_BINARY_DIR}/tests/bin" )
  set( viame_test_working_path "${CMAKE_BINARY_DIR}/tests" )
endif()

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}" )
include_directories( "${CMAKE_SOURCE_DIR}" )
include_directories( "${CMAKE_BINARY_DIR}" )
include_directories( "${CMAKE_SOURCE_DIR}/tests" )
