project(sprokit_python_processes)

set(python_both_arch TRUE)
sprokit_create_python_plugin_init(sprokit/processes)

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(test)
endif ()
