add_custom_target(tooling)

cmake_dependent_option(SPROKIT_ENABLE_CDASH "Enable CDash integration" OFF
  KWIVER_ENABLE_TESTS OFF)

find_program(VALGRIND_EXECUTABLE valgrind)

cmake_dependent_option(SPROKIT_VALGRIND_GENERATE_SUPPRESSIONS "Output suppression rules for valgrind leak detections" OFF
  VALGRIND_EXECUTABLE OFF)
cmake_dependent_option(SPROKIT_VALGRIND_VERBOSE "Make valgrind verbose" OFF
  VALGRIND_EXECUTABLE OFF)
cmake_dependent_option(SPROKIT_VALGRIND_USE_SUPPRESSIONS "Suppress known leaks in valgrind" ON
  VALGRIND_EXECUTABLE OFF)

function (_sprokit_declare_tool_group tool)
  add_custom_target(${tool})
  add_dependencies(tooling
    ${tool})
endfunction ()

if (VALGRIND_EXECUTABLE)
  set(sprokit_valgrind_arguments)

  if (SPROKIT_VALGRIND_GENERATE_SUPPRESSIONS)
    list(APPEND sprokit_valgrind_arguments
      "--gen-suppressions=all")
  endif ()

  if (SPROKIT_VALGRIND_VERBOSE)
    list(APPEND sprokit_valgrind_arguments
      "--verbose")
  endif ()

  if (SPROKIT_VALGRIND_USE_SUPPRESSIONS)
    file(GLOB
      valgrind_suppressions
      "${sprokit_source_dir}/tests/data/valgrind/*.supp")

    foreach (valgrind_suppression IN LISTS valgrind_suppressions)
      list(APPEND sprokit_valgrind_arguments
        "--suppressions=${valgrind_suppression}")
    endforeach ()
  endif ()

  _sprokit_declare_tool_group(valgrind)
  _sprokit_declare_tool_group(cachegrind)
  _sprokit_declare_tool_group(callgrind)
  _sprokit_declare_tool_group(helgrind)
  _sprokit_declare_tool_group(drd)
  _sprokit_declare_tool_group(massif)
  _sprokit_declare_tool_group(dhat)
  _sprokit_declare_tool_group(sgcheck)
  _sprokit_declare_tool_group(bbv)
endif ()

if(NOT MSVC)
  find_program(GPROF_EXECUTABLE gprof)
endif()

if (GPROF_EXECUTABLE)
  _sprokit_declare_tool_group(gprof)
endif ()

function (_sprokit_declare_tool_test tool test)
  add_custom_target(${tool}-${test})
  add_dependencies(${tool}
    ${tool}-${test})
endfunction ()

function (_sprokit_declare_tooled_test test)
  if (VALGRIND_EXECUTABLE)
    _sprokit_declare_tool_test(valgrind ${test})
    _sprokit_declare_tool_test(cachegrind ${test})
    _sprokit_declare_tool_test(callgrind ${test})
    _sprokit_declare_tool_test(helgrind ${test})
    _sprokit_declare_tool_test(drd ${test})
    _sprokit_declare_tool_test(massif ${test})
    _sprokit_declare_tool_test(dhat ${test})
    _sprokit_declare_tool_test(sgcheck ${test})
    _sprokit_declare_tool_test(bbv ${test})
  endif ()
  if (GPROF_EXECUTABLE)
    _sprokit_declare_tool_test(gprof ${test})
  endif ()
endfunction ()

function (sprokit_declare_tooled_test test)
  sprokit_declare_test(${test})
  _sprokit_declare_tooled_test(${test})
endfunction ()

function (sprokit_build_tooled_test test libraries)
  sprokit_build_test(${test} ${libraries} ${ARGN})
  _sprokit_declare_tooled_test(${test})
endfunction ()

function (_sprokit_add_tooled_test tool test instance)
  add_custom_target(${tool}-${test}-${instance})
  add_custom_command(
    TARGET  ${tool}-${test}-${instance}
    COMMAND ${sprokit_test_environment}
            ${${${tool}_args}}
            ${sprokit_test_runner}
            "${sprokit_test_output_path}/test-${test}"
            ${instance}
            ${ARGN}
    WORKING_DIRECTORY
            "${sprokit_test_working_path}"
    COMMENT "Running ${tool} on test \"${test}\" instance \"${instance}\"")
  add_dependencies(${tool}-${test}
    ${tool}-${test}-${instance})
endfunction ()

function (sprokit_add_tooled_test test instance)
  sprokit_add_test(${test} ${instance} ${ARGN})

  if (VALGRIND_EXECUTABLE)
    set(valgrind_args
      "${VALGRIND_EXECUTABLE}"
      --leak-check=full
      --show-reachable=yes
      --track-fds=yes
      --track-origins=yes
      --log-file="${sprokit_test_working_path}/valgrind.log.${test}.${instance}"
      ${sprokit_valgrind_arguments})
    _sprokit_add_tooled_test(valgrind ${test} ${instance} ${ARGN})

    set(cachegrind_args
      "${VALGRIND_EXECUTABLE}"
      --tool=cachegrind
      --log-file="${sprokit_test_working_path}/cachegrind.log.${test}.${instance}"
      --cachegrind-out-file="${sprokit_test_working_path}/cachegrind.out.${test}.${instance}")
    _sprokit_add_tooled_test(cachegrind ${test} ${instance} ${ARGN})

    set(callgrind_args
      "${VALGRIND_EXECUTABLE}"
      --tool=callgrind
      --dump-instr=yes
      --log-file="${sprokit_test_working_path}/callgrind.log.${test}.${instance}"
      --callgrind-out-file="${sprokit_test_working_path}/callgrind.out.${test}.${instance}")
    _sprokit_add_tooled_test(callgrind ${test} ${instance} ${ARGN})

    set(helgrind_args
      "${VALGRIND_EXECUTABLE}"
      --tool=helgrind
      --log-file="${sprokit_test_working_path}/helgrind.log.${test}.${instance}")
    _sprokit_add_tooled_test(helgrind ${test} ${instance} ${ARGN})

    set(drd_args
      "${VALGRIND_EXECUTABLE}"
      --tool=drd
      --log-file="${sprokit_test_working_path}/drd.log.${test}.${instance}")
    _sprokit_add_tooled_test(drd ${test} ${instance} ${ARGN})

    set(massif_args
      "${VALGRIND_EXECUTABLE}"
      --tool=massif
      --stacks=yes
      --log-file="${sprokit_test_working_path}/massif.log.${test}.${instance}"
      --massif-out-file="${sprokit_test_working_path}/massif.out.${test}.${instance}")
    _sprokit_add_tooled_test(massif ${test} ${instance} ${ARGN})

    set(dhat_args
      "${VALGRIND_EXECUTABLE}"
      --tool=exp-dhat
      --log-file="${sprokit_test_working_path}/dhat.log.${test}.${instance}")
    _sprokit_add_tooled_test(dhat ${test} ${instance} ${ARGN})

    set(sgcheck_args
      "${VALGRIND_EXECUTABLE}"
      --tool=exp-ptrcheck
      --log-file="${sprokit_test_working_path}/sgcheck.log.${test}.${instance}")
    _sprokit_add_tooled_test(sgcheck ${test} ${instance} ${ARGN})

    set(bbv_args
      "${VALGRIND_EXECUTABLE}"
      --tool=exp-bbv
      --log-file="${sprokit_test_working_path}/bbv.log.${test}.${instance}"
      --bb-out-file="${sprokit_test_working_path}/bbv.bb.out.${test}.${instance}"
      --pc-out-file="${sprokit_test_working_path}/bbv.pc.log.${test}.${instance}")
    _sprokit_add_tooled_test(bbv ${test} ${instance} ${ARGN})
  endif ()

  if (GPROF_EXECUTABLE)
    set(real_command
      "${sprokit_test_working_path}/test-${test}")
    if (sprokit_test_runner)
      set(real_command
        "${sprokit_test_runner}")
    endif ()

    set(gprof_args)
    _sprokit_add_tooled_test(gprof ${test} ${instance} ${ARGN}
      COMMAND "${GPROF_EXECUTABLE}"
              "${real_command}"
              "${sprokit_test_working_path}/gmon.out"
              > "${sprokit_test_working_path}/gprof.log.${test}.${instance}")
  endif ()
endfunction ()
