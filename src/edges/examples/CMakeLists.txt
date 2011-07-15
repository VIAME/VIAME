project(vistk_edges_examples)

set(examples_srcs
  blocking_pipe_edge.cxx
  dumb_pipe_edge.cxx
  registration.cxx)

add_vistk_plugin(edges_examples
  MAKE_VISTK_EDGES_EXAMPLES_LIB
  ${examples_srcs})
target_link_libraries(edges_examples
  vistk_pipeline)
