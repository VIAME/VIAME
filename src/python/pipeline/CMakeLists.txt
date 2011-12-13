project(vistk_python_pipeline)

add_vistk_python_library(config
  vistk/pipeline
  config.cxx)
target_link_libraries(python-config
  vistk_pipeline
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(edge
  vistk/pipeline
  edge.cxx)
target_link_libraries(python-edge
  vistk_pipeline
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(modules
  vistk/pipeline
  modules.cxx)
target_link_libraries(python-modules
  vistk_pipeline
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(pipeline
  vistk/pipeline
  pipeline.cxx)
target_link_libraries(python-pipeline
  vistk_pipeline
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

make_python_init_file(vistk/pipeline
  config
  edge
  modules
  pipeline)
