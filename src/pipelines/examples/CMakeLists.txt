project(vistk_pipelines_examples)

set(examples_srcs
  registration.cxx
  thread_per_process_pipeline.cxx)

add_vistk_plugin(pipelines_examples
  MAKE_VISTK_PIPELINES_EXAMPLES_LIB
  ${examples_srcs})
target_link_libraries(pipelines_examples
  vistk_pipeline
  ${Boost_THREAD_LIBRARY})
