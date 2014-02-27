project(sprokit_test_pipeline_util)

set(test_libraries
  sprokit_pipeline_util
  sprokit_pipeline
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})

set(sprokit_test_pipelines_directory
  "${sprokit_test_data_directory}/pipelines")

##############################
# Pipeline loading tests
##############################
sprokit_discover_tests(load_pipe test_libraries test_load_pipe.cxx
  "${sprokit_test_pipelines_directory}")

##############################
# Baking tests
##############################
sprokit_discover_tests(pipe_bakery test_libraries test_pipe_bakery.cxx
  "${sprokit_test_pipelines_directory}")

##############################
# Export tests
##############################
sprokit_discover_tests(export_dot test_libraries test_export_dot.cxx
  "${sprokit_test_pipelines_directory}")
