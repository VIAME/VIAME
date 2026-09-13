# VIAME's test helpers
#
# One function survives from kwiver's `kwiver-utils-tests.cmake`, which had
# five. `kwiver_add_test` and `kwiver_discover_tests` drove a test convention
# -- `IMPLEMENT_TEST( name )` macros scanned out of the source with a regular
# expression, one ctest entry per match, `TEST_PROPERTY` comments setting
# ctest properties -- that no VIAME test uses and none has since the import.
# `kwiver_declare_test` built per-group custom targets behind
# `KWIVER_TEST_ADD_TARGETS`, which is not in the cache and so was always off.
#
# What is left is gtest discovery, which every one of the 33 test
# directories uses.

include_guard( GLOBAL )

include( GoogleTest )

#+
# Build a gtest executable and register each of its cases with ctest.
#
#   viame_discover_gtests( group name
#                          [SOURCES ...] [LIBRARIES ...] [ARGUMENTS ...] )
#
# `test_<name>.cxx` unless SOURCES says otherwise. Cases arrive in ctest as
# `<group>:<suite>.<case>`.
#-
function( viame_discover_gtests MODULE NAME )
  cmake_parse_arguments( "" "" "" "SOURCES;LIBRARIES;ARGUMENTS" ${ARGN} )

  if( NOT _SOURCES )
    set( _SOURCES test_${NAME}.cxx )
  endif()

  add_executable( test-${MODULE}-${NAME} ${_SOURCES} )

  set_target_properties( test-${MODULE}-${NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${viame_test_output_path}"
    )

  target_link_libraries( test-${MODULE}-${NAME}
    PRIVATE ${_LIBRARIES} GTest::GTest
    )

  set( extra_args TEST_PREFIX ${MODULE}: DISCOVERY_TIMEOUT 60 )
  if( _ARGUMENTS )
    list( APPEND extra_args EXTRA_ARGS ${_ARGUMENTS} )
  endif()

  # Label them UNIT. Without this a discovered gtest carries no label at all,
  # so `ctest -L UNIT` -- which is how the unit tests are meant to be run --
  # ran none of the two hundred of them. That is how the stereo triangulation
  # regression of finding 1.20 survived: `measurement_utilities_test` had a
  # case that fails on it, and nothing was running that case. PROPERTIES is
  # multi-value, so it goes last.
  list( APPEND extra_args PROPERTIES LABELS UNIT )

  gtest_discover_tests( test-${MODULE}-${NAME} ${extra_args} )
endfunction()
