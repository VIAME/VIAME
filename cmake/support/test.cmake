# Test functions for the sprokit project
# The following functions are defined:
#   sprokit_declare_test
#   sprokit_build_test
#   sprokit_make_test
# Their syntax is:
#   sprokit_declare_test(testname)
#     The first argument is the name of the test group to declare. This adds
#     Group targets for testing, valgrind, callgrind, and gprof targets (if
#     available).
#   sprokit_build_test(testname libraries file1 [file2 ...])
#     The first argument is the name of the test executable to build and the
#     second is the name of a variable containing the libraries that it needs
#     to be linked against. The remaining arguments are the files that are
#     needed to build the test.
#   sprokit_make_test(testname instance [arg1 ...])
#     The first argument is the name of the test and the second is the name of
#     the instance of the test. It creates a target named
#     `test-${testname}-${instance}' which runs the test by itself as well as
#     valgrind, callgrind, and gprof targets (depending on the availability of
#     the tool) that run the test throught the respective tool. Tests also have
#     group targets based on `testname'. The test instance is automatically
#     passed as the first argument to the test.

add_custom_target(tooling)

cmake_dependent_option(SPROKIT_ENABLE_CDASH "Enable CDash integration" OFF
  SPROKIT_ENABLE_TESTING OFF)

option(SPROKIT_ADD_TEST_TARGETS "Add targets for tests to the build system" OFF)
mark_as_advanced(SPROKIT_ADD_TEST_TARGETS)
if (SPROKIT_ADD_TEST_TARGETS)
  add_custom_target(tests)
endif ()

find_program(VALGRIND_EXECUTABLE valgrind)

cmake_dependent_option(SPROKIT_VALGRIND_GENERATE_SUPPRESSIONS "Output suppression rules for valgrind leak detections" OFF
  VALGRIND_EXECUTABLE OFF)
cmake_dependent_option(SPROKIT_VALGRIND_VERBOSE "Make valgrind verbose" OFF
  VALGRIND_EXECUTABLE OFF)
cmake_dependent_option(SPROKIT_VALGRIND_USE_SUPPRESSIONS "Suppress known leaks in valgrind" ON
  VALGRIND_EXECUTABLE OFF)

if (VALGRIND_EXECUTABLE)
  set(sprokit_valgrind_arguments)

  if (SPROKIT_VALGRIND_GENERATE_SUPPRESSIONS)
    set(sprokit_valgrind_arguments
        ${sprokit_valgrind_arguments}
        "--gen-suppressions=all")
  endif ()

  if (SPROKIT_VALGRIND_VERBOSE)
    set(sprokit_valgrind_arguments
      ${sprokit_valgrind_arguments}
      "--verbose")
  endif ()

  if (SPROKIT_VALGRIND_USE_SUPPRESSIONS)
    file(GLOB
      valgrind_suppressions
      "${sprokit_source_dir}/tests/data/valgrind/*.supp")

    foreach (valgrind_suppression ${valgrind_suppressions})
      set(sprokit_valgrind_arguments
        ${sprokit_valgrind_arguments}
        "--suppressions=${valgrind_suppression}")
    endforeach ()
  endif ()

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
endif ()

find_program(GPROF_EXECUTABLE gprof)

if (GPROF_EXECUTABLE)
  add_custom_target(gprof)

  add_dependencies(tooling
    gprof)
endif ()

set(test_base_output_path
  "${sprokit_binary_dir}/bin")
set(test_output_path
  "${test_base_output_path}/${CMAKE_CFG_INTDIR}")
set(test_working_path
  "${sprokit_binary_dir}/tests")

function (sprokit_declare_test testname)
  if (SPROKIT_ADD_TEST_TARGETS)
    add_custom_target(tests-${testname})
    add_dependencies(tests
      tests-${testname})
  endif ()
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
  endif ()
  if (GPROF_EXECUTABLE)
    add_custom_target(gprof-${testname})
    add_dependencies(gprof
      gprof-${testname})
  endif ()
endfunction ()

macro (sprokit_build_test testname libraries)
  add_executable(test-${testname} ${ARGN})
  set_target_properties(test-${testname}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${test_base_output_path}")
  target_link_libraries(test-${testname}
    LINK_PRIVATE
      ${${libraries}})
  sprokit_declare_test(${testname})
endmacro ()

function (sprokit_make_test testname instance)
  if (TARGET test-${testname})
    set(test_path "$<TARGET_FILE:test-${testname}>")
  elseif (CMAKE_CONFIGURATION_TYPES)
    set(test_path "${test_base_output_path}/$<CONFIGURATION>/test-${testname}")
  else ()
    set(test_path "${test_base_output_path}/test-${testname}")
  endif ()

  add_test(
    NAME    test-${testname}-${instance}
    COMMAND ${test_runner}
            "${test_path}"
            ${instance}
            ${ARGN})
  set_tests_properties(test-${testname}-${instance}
    PROPERTIES
      WORKING_DIRECTORY       "${test_working_path}"
      FAIL_REGULAR_EXPRESSION "^Error: ;\nError: ")

  # TODO: How to get CTest the full path to the test with config subdir?
  if (NOT CMAKE_CONFIGURATION_TYPES)
    set_tests_properties(test-${testname}-${instance}
      PROPERTIES
        REQUIRED_FILES "${test_output_path}/test-${testname}")
  endif ()
  if (test_environment)
    set_tests_properties(test-${testname}-${instance}
      PROPERTIES
        ENVIRONMENT ${test_environment})
  endif ()
  if (SPROKIT_ADD_TEST_TARGETS)
    add_custom_target(test-${testname}-${instance})
    add_custom_command(
      TARGET  test-${testname}-${instance}
      COMMAND ${test_environment}
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running test \"${testname}\" instance \"${instance}\"")
    add_dependencies(tests-${testname}
      test-${testname}-${instance})
  endif ()
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
              --log-file="${test_working_path}/valgrind.log.${testname}.${instance}"
              ${sprokit_valgrind_arguments}
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running valgrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(valgrind-${testname}
      valgrind-${testname}-${instance})
    add_custom_target(cachegrind-${testname}-${instance})
    add_custom_command(
      TARGET  cachegrind-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=cachegrind
              --log-file="${test_working_path}/cachegrind.log.${testname}.${instance}"
              --cachegrind-out-file="${test_working_path}/cachegrind.out.${testname}.${instance}"
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
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
              --log-file="${test_working_path}/callgrind.log.${testname}.${instance}"
              --callgrind-out-file="${test_working_path}/callgrind.out.${testname}.${instance}"
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running callgrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(callgrind-${testname}
      callgrind-${testname}-${instance})
    add_custom_target(helgrind-${testname}-${instance})
    add_custom_command(
      TARGET  helgrind-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=helgrind
              --log-file="${test_working_path}/helgrind.log.${testname}.${instance}"
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running helgrind on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(helgrind-${testname}
      helgrind-${testname}-${instance})
    add_custom_target(drd-${testname}-${instance})
    add_custom_command(
      TARGET  drd-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=drd
              --log-file="${test_working_path}/drd.log.${testname}.${instance}"
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
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
              --log-file="${test_working_path}/massif.log.${testname}.${instance}"
              --massif-out-file="${test_working_path}/massif.out.${testname}.${instance}"
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running massif on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(massif-${testname}
      massif-${testname}-${instance})
    add_custom_target(dhat-${testname}-${instance})
    add_custom_command(
      TARGET  dhat-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=exp-dhat
              --log-file="${test_working_path}/dhat.log.${testname}.${instance}"
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running dhat on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(dhat-${testname}
      dhat-${testname}-${instance})
    add_custom_target(sgcheck-${testname}-${instance})
    add_custom_command(
      TARGET  sgcheck-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=exp-ptrcheck
              --log-file="${test_working_path}/sgcheck.log.${testname}.${instance}"
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running sgcheck on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(sgcheck-${testname}
      sgcheck-${testname}-${instance})
    add_custom_target(bbv-${testname}-${instance})
    add_custom_command(
      TARGET  bbv-${testname}-${instance}
      COMMAND ${test_environment}
              "${VALGRIND_EXECUTABLE}"
              --tool=exp-bbv
              --log-file="${test_working_path}/bbv.log.${testname}.${instance}"
              --bb-out-file="${test_working_path}/bbv.bb.out.${testname}.${instance}"
              --pc-out-file="${test_working_path}/bbv.pc.log.${testname}.${instance}"
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running bbv on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(bbv-${testname}
      bbv-${testname}-${instance})
  endif ()
  if (GPROF_EXECUTABLE)
    set(real_command
      "${test_working_path}/test-${testname}")
    if (test_runner)
      set(real_command
        ${test_runner})
    endif ()

    add_custom_target(gprof-${testname}-${instance})
    add_custom_command(
      TARGET  gprof-${testname}-${instance}
      COMMAND ${test_environment}
              ${test_runner}
              "${test_output_path}/test-${testname}"
              ${instance}
              ${ARGN}
      COMMAND "${GPROF_EXECUTABLE}"
              "${real_command}"
              "${test_working_path}/gmon.out"
              > "${test_working_path}/gprof.log.${testname}.${instance}"
      WORKING_DIRECTORY
              "${test_working_path}"
      COMMENT "Running gprof on test \"${testname}\" instance \"${instance}\"")
    add_dependencies(gprof-${testname}
      gprof-${testname}-${instance})
  endif ()
endfunction ()
