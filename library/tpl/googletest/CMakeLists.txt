# GoogleTest, fetched and built when tests are enabled
#
# Not vendored, on the user's reasoning: only tests link it and tests are not
# in a release, so carrying its source would add to the repository something
# that never ships. Not found either, which is what tied the build to
# fletch -- `find_package( GTest REQUIRED )` in the test setup
# resolved through fletch's prefix and was the last thing that did.
#
# So it is downloaded at configure time and built with the tree, once, and
# only when `VIAME_ENABLE_TESTS` is on. A build with tests off never reaches
# this file.
#
# Offline builds: set `VIAME_GTEST_SOURCE_DIR` to a googletest checkout and
# it is used instead of the download. `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`,
# CMake's own escape hatch, works too.

include( FetchContent )

set( VIAME_GTEST_VERSION "1.14.0" CACHE STRING
     "GoogleTest version to build for the tests" )
mark_as_advanced( VIAME_GTEST_VERSION )

set( VIAME_GTEST_SOURCE_DIR "" CACHE PATH
     "An existing googletest checkout to build instead of downloading one" )
mark_as_advanced( VIAME_GTEST_SOURCE_DIR )

if( VIAME_GTEST_SOURCE_DIR )
  set( FETCHCONTENT_SOURCE_DIR_GOOGLETEST "${VIAME_GTEST_SOURCE_DIR}" )
endif()

FetchContent_Declare( googletest
  URL https://github.com/google/googletest/archive/refs/tags/v${VIAME_GTEST_VERSION}.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

# Its own tests and install rules are not wanted; VIAME ships no gtest
set( BUILD_GMOCK OFF CACHE BOOL "" FORCE )
set( INSTALL_GTEST OFF CACHE BOOL "" FORCE )
set( gtest_force_shared_crt ON CACHE BOOL "" FORCE )

FetchContent_MakeAvailable( googletest )

# `GTest::GTest` is what `FindGTest` called the library and what
# `kwiver-utils-tests.cmake` links; googletest's own CMake calls it
# `GTest::gtest`. Both names are used across this tree, so both exist.
if( TARGET gtest AND NOT TARGET GTest::GTest )
  add_library( GTest::GTest ALIAS gtest )
endif()

if( TARGET gtest_main AND NOT TARGET GTest::Main )
  add_library( GTest::Main ALIAS gtest_main )
endif()
