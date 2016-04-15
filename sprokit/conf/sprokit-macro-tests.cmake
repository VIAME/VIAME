# Test functions for the sprokit project
# The following functions are defined:
#
#   sprokit_declare_test
#   sprokit_build_test
#   sprokit_add_test
#
# The following variables may be used to control the behavior of the functions:
#
#   SPROKIT_TEST_ADD_TARGETS
#     A boolean flag which, if true, adds targets to the build to run the tests
#     in groupings. This is added to the cache as an advanced variable.
#
#   sprokit_test_output_path
#     Where to place test binaries and expect to find them. This must be set.
#
#   sprokit_test_working_path
#     The directory to run tests in. This must be set.
#
#   sprokit_test_runner
#     A top-level executable (possibly with arguments) to run the main
#     test-name executable under. As an example, for any tests which are
#     Python, this should be set to ${PYTHON_EXECUTABLE} since Python files by
#     themselves are not executable on all platforms.
#
# Their syntax is:
#
#   sprokit_declare_test(name)
#     Declares a test grouping. Use this when a test is not built using
#     sprokit_build_test.
#
#   sprokit_build_test(name libraryvar [source ...])
#     Builds a test and declares the test as well. The library passed as
#     libraryvar should contain the list of libraries to link.
#
#   sprokit_add_test(name instance [arg ...])
#     Adds a test to run. This runs the executable test-${name} with the
#     arguments ${instance} ${ARGN}. If enabled, it adds a target named
#     test-${name}-${instance} to be run by the build if wanted.

option(SPROKIT_TEST_ADD_TARGETS "Add targets for tests to the build system" OFF)
mark_as_advanced(SPROKIT_TEST_ADD_TARGETS)
if (SPROKIT_TEST_ADD_TARGETS)
  add_custom_target(tests)
endif ()

function (sprokit_declare_test name)
  if (NOT SPROKIT_TEST_ADD_TARGETS)
    return()
  endif ()

  add_custom_target("tests-${name}")
  add_dependencies(tests
    "tests-${name}")
endfunction ()

function (sprokit_build_test name libraries)
  add_executable(test-${name} ${ARGN})
  set_target_properties(test-${name}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${sprokit_test_output_path}")
  target_link_libraries(test-${name}
    LINK_PRIVATE
      ${${libraries}})
  sprokit_declare_test(${name})
endfunction ()

function (sprokit_add_test name instance)
  if (TARGET test-${name})
    set(test_path "$<TARGET_FILE:test-${name}>")
  elseif (CMAKE_CONFIGURATION_TYPES)
    set(test_path "${sprokit_test_output_path}/$<CONFIGURATION>/test-${name}")
  else ()
    set(test_path "${sprokit_test_output_path}/test-${name}")
  endif ()

  add_test(
    NAME    test-${name}-${instance}
    COMMAND ${sprokit_test_runner}
            "${test_path}"
            ${instance}
            ${ARGN})
  set_tests_properties(test-${name}-${instance}
    PROPERTIES
      FAIL_REGULAR_EXPRESSION "^Error: ;\nError: ")
  if (sprokit_test_working_path)
    set_tests_properties(test-${name}-${instance}
      PROPERTIES
        WORKING_DIRECTORY       "${sprokit_test_working_path}")
  endif ()

  # TODO: How to get CTest the full path to the test with config subdir?
  if (NOT CMAKE_CONFIGURATION_TYPES)
    set_tests_properties(test-${name}-${instance}
      PROPERTIES
      REQUIRED_FILES "${sprokit_test_output_path}/${CMAKE_CFG_INTDIR}/test-${name}")
  endif ()
  if (sprokit_test_environment)
    set_tests_properties(test-${name}-${instance}
      PROPERTIES
        ENVIRONMENT "${sprokit_test_environment}")
  endif ()
  if (SPROKIT_TEST_ADD_TARGETS)
    add_custom_target(test-${name}-${instance})
    add_custom_command(
      TARGET  test-${name}-${instance}
      COMMAND ${sprokit_test_environment}
              ${sprokit_test_runner}
              "${sprokit_test_output_path}/${CMAKE_CFG_INTDIR}/test-${name}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${sprokit_test_working_path}"
      COMMENT "Running test \"${name}\" instance \"${instance}\"")
    add_dependencies(tests-${name}
      test-${name}-${instance})
  endif ()
endfunction ()
