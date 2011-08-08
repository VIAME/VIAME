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

add_vistk_python_library(process_registry
  vistk/pipeline
  process_registry.cxx)
target_link_libraries(python-process_registry
  vistk_pipeline
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(schedule
  vistk/pipeline
  schedule.cxx)
target_link_libraries(python-schedule
  vistk_pipeline
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(schedule_registry
  vistk/pipeline
  schedule_registry.cxx)
target_link_libraries(python-schedule_registry
  vistk_pipeline
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

make_python_init_file(vistk/pipeline
  config
  edge
  modules
  pipeline
  process_registry
  schedule
  schedule_registry)
