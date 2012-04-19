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

add_custom_target(tooling)

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
      "--suppressions=${CMAKE_SOURCE_DIR}/tests/data/valgrind/glibc.supp")
  endif (VISTK_VALGRIND_USE_SUPPRESSIONS)

  add_custom_target(tests)
  add_custom_target(valgrind)
  add_custom_target(cachegrind)
  add_custom_target(callgrind)
  add_custom_target(helgrind)
  add_custom_target(drd)
  add_custom_target(massif)
  add_custom_target(dhat)
  add_custom_target(sgcheck)
  add_custom_target(bbv)

  add_dependencies(tooling
    valgrind
    cachegrind
    callgrind
    helgrind
    drd
    massif
    dhat
    sgcheck
    bbv)
endif (VALGRIND_EXECUTABLE)

find_program(GPROF_EXECUTABLE gprof)

if (GPROF_EXECUTABLE)
  add_custom_target(gprof)

  add_dependencies(tooling
    gprof)
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
    add_dependencies(tests
      tests-${testname})
  endif (NOT WIN32)
  if (VALGRIND_EXECUTABLE)
    add_custom_target(valgrind-${testname})
    add_custom_target(cachegrind-${testname})
    add_custom_target(callgrind-${testname})
    add_custom_target(helgrind-${testname})
    add_custom_target(drd-${testname})
    add_custom_target(massif-${testname})
    add_custom_target(dhat-${testname})
    add_custom_target(sgcheck-${testname})
    add_custom_target(bbv-${testname})
    add_dependencies(valgrind
      valgrind-${testname})
    add_dependencies(cachegrind
      cachegrind-${testname})
    add_dependencies(callgrind
      callgrind-${testname})
    add_dependencies(helgrind
      helgrind-${testname})
    add_dependencies(drd
      drd-${testname})
    add_dependencies(massif
      massif-${testname})
    add_dependencies(dhat
      dhat-${testname})
    add_dependencies(sgcheck
      sgcheck-${testname})
    add_dependencies(bbv
      bbv-${testname})
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
    COMMAND ${test_runner}
            "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
            ${instance}
            ${ARGN})
  set_tests_properties(test-${testname}-${instance}
    PROPERTIES
      WORKING_DIRECTORY       "${EXECUTABLE_OUTPUT_PATH}"
      FAIL_REGULAR_EXPRESSION "^Error: ")
  if (test_environment)
    set_tests_properties(test-${testname}-${instance}
      PROPERTIES
        ENVIRONMENT ${test_environment})
  endif (test_environment)
  if (NOT WIN32)
    add_custom_target(test-${testname}-${instance})
    add_custom_command(
      TARGET  test-${testname}-${instance}
      COMMAND ${test_environment}
              ${test_runner}
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
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --leak-check=full
              --show-reachable=yes
              --track-fds=yes
              --track-origins=yes
              --log-file="${EXECUTABLE_OUTPUT_PATH}/valgrind.log.${testname}.${instance}"
              ${vistk_valgrind_arguments}
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running valgrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(valgrind-${testname}
      valgrind-${testname}-${instance})
    add_custom_target(cachegrind-${testname}-${instance})
    add_custom_command(
      TARGET  cachegrind-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=cachegrind
              --log-file="${EXECUTABLE_OUTPUT_PATH}/cachegrind.log.${testname}.${instance}"
              --cachegrind-out-file="${EXECUTABLE_OUTPUT_PATH}/cachegrind.out.${testname}.${instance}"
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running cachegrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(cachegrind-${testname}
      cachegrind-${testname}-${instance})
    add_custom_target(callgrind-${testname}-${instance})
    add_custom_command(
      TARGET  callgrind-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=callgrind
              --dump-instr=yes
              --log-file="${EXECUTABLE_OUTPUT_PATH}/callgrind.log.${testname}.${instance}"
              --callgrind-out-file="${EXECUTABLE_OUTPUT_PATH}/callgrind.out.${testname}.${instance}"
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running callgrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(callgrind-${testname}
      callgrind-${testname}-${instance})
    add_custom_target(helgrind-${testname}-${instance})
    add_custom_command(
      TARGET  helgrind-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=helgrind
              --log-file="${EXECUTABLE_OUTPUT_PATH}/helgrind.log.${testname}.${instance}"
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running helgrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(helgrind-${testname}
      helgrind-${testname}-${instance})
    add_custom_target(drd-${testname}-${instance})
    add_custom_command(
      TARGET  drd-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=drd
              --log-file="${EXECUTABLE_OUTPUT_PATH}/drd.log.${testname}.${instance}"
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running drd on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(drd-${testname}
      drd-${testname}-${instance})
    add_custom_target(massif-${testname}-${instance})
    add_custom_command(
      TARGET  massif-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=massif
              --stacks=yes
              --log-file="${EXECUTABLE_OUTPUT_PATH}/massif.log.${testname}.${instance}"
              --massif-out-file="${EXECUTABLE_OUTPUT_PATH}/massif.out.${testname}.${instance}"
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running massif on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(massif-${testname}
      massif-${testname}-${instance})
    add_custom_target(dhat-${testname}-${instance})
    add_custom_command(
      TARGET  dhat-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=exp-dhat
              --log-file="${EXECUTABLE_OUTPUT_PATH}/dhat.log.${testname}.${instance}"
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running dhat on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(dhat-${testname}
      dhat-${testname}-${instance})
    add_custom_target(sgcheck-${testname}-${instance})
    add_custom_command(
      TARGET  sgcheck-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=exp-ptrcheck
              --log-file="${EXECUTABLE_OUTPUT_PATH}/sgcheck.log.${testname}.${instance}"
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running sgcheck on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(sgcheck-${testname}
      sgcheck-${testname}-${instance})
    add_custom_target(bbv-${testname}-${instance})
    add_custom_command(
      TARGET  bbv-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=exp-bbv
              --log-file="${EXECUTABLE_OUTPUT_PATH}/bbv.log.${testname}.${instance}"
              --bb-out-file="${EXECUTABLE_OUTPUT_PATH}/bbv.bb.out.${testname}.${instance}"
              --pc-out-file="${EXECUTABLE_OUTPUT_PATH}/bbv.pc.log.${testname}.${instance}"
              ${test_runner}
              "${EXECUTABLE_OUTPUT_PATH}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${TEST_WORKING_DIRECTORY}"
      COMMENT "Running bbv on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(bbv-${testname}
      bbv-${testname}-${instance})
  endif (VALGRIND_EXECUTABLE)
  if (GPROF_EXECUTABLE)
    set(real_command
      "${EXECUTABLE_OUTPUT_PATH}/test-${testname}")
    if (test_runner)
      set(real_command
        ${test_runner})
    endif (test_runner)

    add_custom_target(gprof-${testname}-${instance})
    add_custom_command(
      TARGET  gprof-${testname}-${instance}
      COMMAND ${test_environment}
              ${test_runner}
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
