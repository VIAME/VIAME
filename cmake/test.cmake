# Test functions for the vistk project
# The following functions are defined:
#   vistk_declare_test
#   vistk_build_test
#   vistk_make_test
# Their syntax is:
#   vistk_declare_test(testname)
#     The first argument is the name of the test group to declare. This adds
#     Group targets for testing, valgrind, callgrind, and gprof targets (if
#     available).
#   vistk_build_test(testname libraries file1 [file2 ...])
#     The first argument is the name of the test executable to build and the
#     second is the name of a variable containing the libraries that it needs
#     to be linked against. The remaining arguments are the files that are
#     needed to build the test.
#   vistk_make_test(testname instance [arg1 ...])
#     The first argument is the name of the test and the second is the name of
#     the instance of the test. It creates a target named
#     `test-${testname}-${instance}' which runs the test by itself as well as
#     valgrind, callgrind, and gprof targets (depending on the availability of
#     the tool) that run the test throught the respective tool. Tests also have
#     group targets based on `testname'. The test instance is automatically
#     passed as the first argument to the test.

enable_testing()
include(CTest)

find_program(VALGRIND_EXECUTABLE valgrind)

if (VALGRIND_EXECUTABLE)
  option(VISTK_VALGRIND_GENERATE_SUPPRESSIONS "Output suppression rules for valgrind leak detections" OFF)
  option(VISTK_VALGRIND_VERBOSE "Make valgrind verbose" OFF)
  option(VISTK_VALGRIND_USE_SUPPRESSIONS "Suppress known leaks in valgrind" ON)

  set(vistk_valgrind_arguments)
  if (VISTK_VALGRIND_GENERATE_SUPPRESSIONS)
    set(vistk_valgrind_arguments
        ${vistk_valgrind_arguments}
        "--gen-suppressions=all")
  endif (VISTK_VALGRIND_GENERATE_SUPPRESSIONS)
  if (VISTK_VALGRIND_VERBOSE)
    set(vistk_valgrind_arguments
      ${vistk_valgrind_arguments}
      "--verbose")
  endif (VISTK_VALGRIND_VERBOSE)
  if (VISTK_VALGRIND_USE_SUPPRESSIONS)
    set(vistk_valgrind_arguments
      ${vistk_valgrind_arguments}
      "--suppressions=${CMAKE_SOURCE_DIR}/tests/valgrind/glibc.supp")
  endif (VISTK_VALGRIND_USE_SUPPRESSIONS)

  add_custom_target(valgrind)
  add_custom_target(callgrind)
  add_custom_target(ptrcheck)
endif (VALGRIND_EXECUTABLE)

find_program(GPROF_EXECUTABLE gprof)

if (GPROF_EXECUTABLE)
  add_custom_target(gprof)
endif (GPROF_EXECUTABLE)

set(TEST_WORKING_DIRECTORY
  "${EXECUTABLE_OUTPUT_PATH}")

if (WIN32)
  set(TEST_WORKING_DIRECTORY
    "${TEST_WORKING_DIRECTORY}/$<CONFIGURATION>")
endif (WIN32)

set(BUILDNAME "" CACHE STRING "The build name for CDash submissions")

function (vistk_declare_test testname)
  if (NOT WIN32)
    add_custom_target(tests-${testname})
  endif (NOT WIN32)
  if (VALGRIND_EXECUTABLE)
    add_custom_target(valgrind-${testname})
    add_custom_target(callgrind-${testname})
    add_custom_target(ptrcheck-${testname})
    add_dependencies(valgrind
      valgrind-${testname})
    add_dependencies(callgrind
      callgrind-${testname})
    add_dependencies(ptrcheck
      ptrcheck-${testname})
  endif (VALGRIND_EXECUTABLE)
  if (GPROF_EXECUTABLE)
    add_custom_target(gprof-${testname})
    add_dependencies(gprof
      gprof-${testname})
  endif (GPROF_EXECUTABLE)
endfunction (vistk_declare_test)

macro (vistk_build_test testname libraries)
  add_executable(test-${testname} ${ARGN})
  target_link_libraries(test-${testname}
    ${${libraries}})
  vistk_declare_test(${testname})
endmacro (vistk_build_test)

function (vistk_make_test testname instance)
  add_test(NAME test-${testname}-${instance}
    COMMAND ${TEST_RUNNER}
            "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
            ${instance}
            ${ARGN})
  set_tests_properties(test-${testname}-${instance}
    PROPERTIES
      WORKING_DIRECTORY       "${EXECUTABLE_OUTPUT_PATH}"
      FAIL_REGULAR_EXPRESSION "Error: ")
  if (NOT WIN32)
    add_custom_target(test-${testname}-${instance})
    add_custom_command(
      TARGET  test-${testname}-${instance}
      COMMAND ${TEST_RUNNER}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${EXECUTABLE_OUTPUT_PATH}"
      COMMENT "Running test \"${testname}\" instance \"${instance}\"")
    add_dependencies(tests-${testname}
      test-${testname}-${instance})
  endif (NOT WIN32)
  if (VALGRIND_EXECUTABLE)
    add_custom_target(valgrind-${testname}-${instance})
    add_custom_command(
      TARGET  valgrind-${testname}-${instance}
      COMMAND "${VALGRIND_EXECUTABLE}"
              --leak-check=full
              --show-reachable=yes
              --track-fds=yes
              --track-origins=yes
              --log-file="${EXECUTABLE_OUTPUT_PATH}/valgrind.log.${testname}.${instance}"
              ${vistk_valgrind_arguments}
              ${TEST_RUNNER}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running valgrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(valgrind-${testname}
      valgrind-${testname}-${instance})
    add_custom_target(callgrind-${testname}-${instance})
    add_custom_command(
      TARGET  callgrind-${testname}-${instance}
      COMMAND "${VALGRIND_EXECUTABLE}"
              --tool=callgrind
              --dump-instr=yes
              --log-file="${EXECUTABLE_OUTPUT_PATH}/callgrind.log.${testname}.${instance}"
              --callgrind-out-file="${EXECUTABLE_OUTPUT_PATH}/callgrind.out.${testname}.${instance}"
              ${TEST_RUNNER}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running callgrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(callgrind-${testname}
      callgrind-${testname}-${instance})
    add_custom_target(ptrcheck-${testname}-${instance})
    add_custom_command(
      TARGET  ptrcheck-${testname}-${instance}
      COMMAND "${VALGRIND_EXECUTABLE}"
              --tool=exp-ptrcheck
              --log-file="${EXECUTABLE_OUTPUT_PATH}/ptrcheck.log.${testname}.${instance}"
              ${TEST_RUNNER}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running ptrcheck on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(ptrcheck-${testname}
      ptrcheck-${testname}-${instance})
  endif (VALGRIND_EXECUTABLE)
  if (GPROF_EXECUTABLE)
    set(real_command
      "${EXECUTABLE_OUTPUT_PATH}/test-${testname}")
    if (TEST_RUNNER)
      set(real_command
        ${TEST_RUNNER})
    endif (TEST_RUNNER)

    add_custom_target(gprof-${testname}-${instance})
    add_custom_command(
      TARGET  gprof-${testname}-${instance}
      COMMAND ${TEST_RUNNER}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      COMMAND "${GPROF_EXECUTABLE}"
              "${real_command}"
              "${EXECUTABLE_OUTPUT_PATH}/gmon.out"
              > "${EXECUTABLE_OUTPUT_PATH}/gprof.log.${testname}.${instance}"
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running gprof on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(gprof-${testname}
      gprof-${testname}-${instance})
  endif (GPROF_EXECUTABLE)
endfunction (vistk_make_test)
