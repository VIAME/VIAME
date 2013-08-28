project(sprokit_test_pipeline)

set(test_libraries
  sprokit_pipeline)

##############################
# Config tests
##############################
sprokit_discover_tests(config test_libraries test_config.cxx)

##############################
# Datum tests
##############################
sprokit_discover_tests(datum test_libraries test_datum.cxx)

##############################
# Stamp tests
##############################
sprokit_discover_tests(stamp test_libraries test_stamp.cxx)

##############################
# Edge tests
##############################
set(edge_libraries
  ${test_libraries}
  ${Boost_THREAD_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_CHRONO_LIBRARY})

sprokit_discover_tests(edge edge_libraries test_edge.cxx)

##############################
# Modules tests
##############################
sprokit_add_plugin(processes_test SHARED
  processes_test.cxx)
target_link_libraries(processes_test
  LINK_PRIVATE
    sprokit_pipeline)

set_target_properties(processes_test
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    PREFIX                   ""
    SUFFIX                   ${CMAKE_SHARED_MODULE_SUFFIX})

sprokit_discover_tests(modules test_libraries test_modules.cxx)

##############################
# Process tests
##############################
sprokit_discover_tests(process test_libraries test_process.cxx)

##############################
# Process cluster tests
##############################
sprokit_discover_tests(process_cluster test_libraries test_process_cluster.cxx)

##############################
# Scheduler tests
##############################
sprokit_discover_tests(scheduler test_libraries test_scheduler.cxx)

##############################
# Pipeline tests
##############################
sprokit_discover_tests(pipeline test_libraries test_pipeline.cxx)

##############################
# Introspection tests
##############################
sprokit_build_tooled_test(process_introspection test_libraries test_process_introspection.cxx)

set(test_python_path)

if (SPROKIT_ENABLE_PYTHON)
  set(python_module_path
    "${sprokit_python_output_path}/${CMAKE_CFG_INTDIR}")

  set(cur_python_path
    "$ENV{PYTHONPATH}")

  if (cur_python_path)
    set(cur_python_path
      ":${cur_python_path}")
  endif ()

  set(test_python_path
    "${python_module_path}${cur_python_path}")
endif ()

# TODO C++ exceptions get lost in the Python glue code.
#set(sprokit_test_environment "PYTHONPATH=${test_python_path}")
sprokit_add_tooled_test(process_introspection all)
unset(sprokit_test_environment)

##############################
# Process registry tests
##############################
sprokit_discover_tests(process_registry test_libraries test_process_registry.cxx)

##############################
# Scheduler registry tests
##############################
sprokit_discover_tests(scheduler_registry test_libraries test_scheduler_registry.cxx)

##############################
# Running tests
##############################
sprokit_build_tooled_test(run test_libraries test_run.cxx)

set(schedulers
  sync
  thread_per_process)

if (SPROKIT_ENABLE_PYTHON)
  list(APPEND schedulers
    pythread_per_process)
endif ()

function (sprokit_add_tooled_run_test group instance)
  foreach (scheduler IN LISTS schedulers)
    if (SPROKIT_ENABLE_PYTHON)
      if (scheduler STREQUAL "pythread_per_process" OR
          instance STREQUAL "pysimple_pipeline")
        set(sprokit_test_environment "PYTHONPATH=${test_python_path}")
      endif ()
    endif ()

    sprokit_add_tooled_test(${group} ${instance}-${scheduler})

    set_tests_properties(test-${group}-${instance}-${scheduler}
      PROPERTIES
        TIMEOUT 5)

    unset(sprokit_test_environment)
  endforeach ()
endfunction ()

sprokit_add_tooled_run_test(run simple_pipeline)
if (SPROKIT_ENABLE_PYTHON)
  sprokit_add_tooled_run_test(run pysimple_pipeline)
endif ()
sprokit_add_tooled_run_test(run multiplier_pipeline)
sprokit_add_tooled_run_test(run multiplier_cluster_pipeline)
