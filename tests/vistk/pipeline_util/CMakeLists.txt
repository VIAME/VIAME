project(vistk_test_pipeline_util)

set(libraries
  vistk_pipeline_util
  vistk_pipeline)

set(vistk_test_pipelines_directory
  ${vistk_test_data_directory}/pipelines)

##############################
# Pipeline loading tests
##############################
set(load_pipe_libraries
  ${libraries}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY})
vistk_build_test(load_pipe load_pipe_libraries test_load_pipe.cxx)
