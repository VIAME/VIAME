# Test functions for the kwiver project
# The following functions are defined:
#
#   kwiver_declare_test
#   kwiver_build_test
#   kwiver_add_test
#   kwiver_discover_tests
#   kwiver_discover_gtests
#
# The following variables may be used to control the behavior of the functions:
#
#   KWIVER_TEST_ADD_TARGETS
#     A boolean flag which, if true, adds targets to the build to run the tests
#     in groupings. This is added to the cache as an advanced variable.
#
#   kwiver_test_output_path
#     Where to place test binaries and expect to find them. This must be set.
#
#   kwiver_test_working_path
#     The directory to run tests in. This must be set.
#
#   kwiver_test_runner
#     A top-level executable (possibly with arguments) to run the main
#     test-name executable under. As an example, for any tests which are
#     Python, this should be set to ${PYTHON_EXECUTABLE} since Python files by
#     themselves are not executable on all platforms. For c++ test this symbol
#     is left empty since the test is already executable.
#
#
# Their syntax is:
#
#   kwiver_declare_test(name)
#     Declares a test grouping. Use this when a test is not built using
#     kwiver_build_test.
#
#   kwiver_build_test(name libraryvar [source ...])
#     Builds a test and declares the test as well. The library passed as
#     libraryvar should contain the list of libraries to link.
#
#   kwiver_add_test(name instance [arg ...])
#     Adds a test to run. This runs the executable test-${name} with the
#     arguments ${instance} ${ARGN}. If enabled, it adds a target named
#     test-${name}-${instance} to be run by the build if wanted.
#
#   (RECOMMENDED)
#   kwiver_discover_tests(group libraries file [arg1 [arg2 ...]])
#     Discovers tests declared within the specified ``file`` (test names must
#     be alphanumeric), defining a test target executable that under the given
#     ``group`` name. The executable generated will be linked against the
#     given ``libraries``. Additional arguments and are eventually passed to
#     the ``add_test()`` call under the hood.
#
#   (RECOMMENDED)
#   kwiver_discover_gtests(group name [SOURCES source ...]
#                         [LIBRARIES library ...]
#                         [ARGUMENTS arg ...])
#     Build a test executable that uses Google Test. Tests are discovered and
#     registered automatically. If no sources are listed, guesses that the
#     source file is named ``test_<name>.cxx``. ``LIBRARIES`` may be used to
#     specify a list of libraries that the executable needs to link. Test names
#     will be prefixed with ``group``.
#

include(GoogleTest)

option(KWIVER_TEST_ADD_TARGETS "Add targets for tests to the build system" OFF)
mark_as_advanced(KWIVER_TEST_ADD_TARGETS)
if (KWIVER_TEST_ADD_TARGETS)
  add_custom_target(tests)
  add_dependencies(tests gtest_discover_tests)
endif ()

# ------------------------------------------------------------------
function (kwiver_declare_test name)
  if (NOT KWIVER_TEST_ADD_TARGETS)
    return()
  endif ()

  add_custom_target("tests-${name}")
  add_dependencies(tests
    "tests-${name}")
endfunction ()

# ------------------------------------------------------------------
function (kwiver_build_test name libraries)
  add_executable(test-${name} ${ARGN})
  set_target_properties(test-${name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${kwiver_test_output_path}")
  target_link_libraries(test-${name}
    LINK_PRIVATE
      ${${libraries}})
  kwiver_declare_test(${name})
endfunction ()

# ------------------------------------------------------------------
function (kwiver_add_test name instance)
  if (TARGET test-${name})
    set(test_path "$<TARGET_FILE:test-${name}>")
  elseif (CMAKE_CONFIGURATION_TYPES)
    set(test_path "${kwiver_test_output_path}/$<CONFIGURATION>/test-${name}")
  else ()
    set(test_path "${kwiver_test_output_path}/test-${name}")
  endif ()

  add_test(
    NAME    test-${name}-${instance}
    COMMAND ${kwiver_test_runner}
            "${test_path}"
            ${instance}
            ${ARGN})
  set_tests_properties(test-${name}-${instance}
    PROPERTIES
      FAIL_REGULAR_EXPRESSION "^Error: ;\nError: ")
  if (kwiver_test_working_path)
    set_tests_properties(test-${name}-${instance}
      PROPERTIES
        WORKING_DIRECTORY       "${kwiver_test_working_path}")
  endif ()

  # TODO: How to get CTest the full path to the test with config subdir?
  if (NOT CMAKE_CONFIGURATION_TYPES)
#    set_tests_properties(test-${name}-${instance}
#      PROPERTIES
#      REQUIRED_FILES "${kwiver_test_output_path}/${CMAKE_CFG_INTDIR}/test-${name}")
  endif ()

  if (kwiver_test_environment)
    set_tests_properties(test-${name}-${instance}
      PROPERTIES
        ENVIRONMENT "${kwiver_test_environment}")
  endif ()

  if (KWIVER_TEST_ADD_TARGETS)
    add_custom_target(test-${name}-${instance})
    add_custom_command(
      TARGET  test-${name}-${instance}
      COMMAND ${kwiver_test_environment}
              ${kwiver_test_runner}
              "${kwiver_test_output_path}/${CMAKE_CFG_INTDIR}/test-${name}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${kwiver_test_working_path}"
      COMMENT "Running test \"${name}\" instance \"${instance}\"")
    add_dependencies(tests-${name}
      test-${name}-${instance})
  endif ()
endfunction ()

# ------------------------------------------------------------------
function (kwiver_discover_tests group libraries file)
  file(STRINGS "${file}" test_lines)
  set(properties)

  kwiver_build_test("${group}" "${libraries}" "${file}")

  foreach (test_line IN LISTS test_lines)
    set(test_name)
    set(property)

    # Note that this does not handle comments well
    string(REGEX MATCH "^IMPLEMENT_TEST\\( *([A-Za-z_0-9]+) *\\)$"
      match "${test_line}")
    if (match)
      set(test_name "${CMAKE_MATCH_1}")
      kwiver_add_test("${group}" "${test_name}" ${ARGN})
      if (properties)
        set_tests_properties("test-${group}-${test_name}"
          PROPERTIES
            ${properties})
      endif ()
      set(properties)
      set(kwiver_test_environment)
    endif ()
    string(REGEX MATCHALL "^TEST_PROPERTY\\( *([A-Za-z_0-9]+) *, *(.*) *\\)$"
      match "${test_line}")
    if (match)
      set(prop "${CMAKE_MATCH_1}")
      string(CONFIGURE "${CMAKE_MATCH_2}" prop_value
        @ONLY)
      if (prop STREQUAL "ENVIRONMENT")
        set(kwiver_test_environment
          "${prop_value}")
      else ()
        set(property "${prop}" "${prop_value}")
        list(APPEND properties
          "${property}")
      endif ()
    endif ()
  endforeach ()
endfunction ()

# -----------------------------------------------------------------------------
function (kwiver_discover_gtests MODULE NAME)
  cmake_parse_arguments("" "" "" "SOURCES;LIBRARIES;ARGUMENTS" ${ARGN})
  if (NOT _SOURCES)
    set(_SOURCES test_${NAME}.cxx)
  endif()
  list(APPEND _LIBRARIES GTest::GTest)

  set(EXTRA_ARGS TEST_PREFIX ${MODULE}:)
  if (_ARGUMENTS)
    list(APPEND EXTRA_ARGS EXTRA_ARGS ${_ARGUMENTS})
  endif()

  kwiver_build_test(${MODULE}-${NAME} _LIBRARIES ${_SOURCES})
  gtest_discover_tests(test-${MODULE}-${NAME} ${EXTRA_ARGS})
endfunction()
