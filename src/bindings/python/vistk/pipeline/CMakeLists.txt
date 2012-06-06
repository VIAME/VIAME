project(vistk_python_vistk_pipeline)

add_vistk_python_library(config
  vistk/pipeline
  config.cxx)
target_link_libraries(python-vistk.pipeline-config
  vistk_pipeline
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(datum
  vistk/pipeline
  datum.cxx)
target_link_libraries(python-vistk.pipeline-datum
  vistk_pipeline
  vistk_python_any_conversion
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(edge
  vistk/pipeline
  edge.cxx)
target_link_libraries(python-vistk.pipeline-edge
  vistk_pipeline
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(modules
  vistk/pipeline
  modules.cxx)
target_link_libraries(python-vistk.pipeline-modules
  vistk_pipeline
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(pipeline
  vistk/pipeline
  pipeline.cxx)
target_link_libraries(python-vistk.pipeline-pipeline
  vistk_pipeline
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(process
  vistk/pipeline
  process.cxx)
target_link_libraries(python-vistk.pipeline-process
  vistk_pipeline
  vistk_python_util
  python_exceptions
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(process_registry
  vistk/pipeline
  process_registry.cxx)
target_link_libraries(python-vistk.pipeline-process_registry
  vistk_pipeline
  vistk_python_util
  python_threading
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(schedule
  vistk/pipeline
  schedule.cxx)
target_link_libraries(python-vistk.pipeline-schedule
  vistk_pipeline
  vistk_python_util
  python_exceptions
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(schedule_registry
  vistk/pipeline
  schedule_registry.cxx)
target_link_libraries(python-vistk.pipeline-schedule_registry
  vistk_pipeline
  vistk_python_util
  python_threading
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(stamp
  vistk/pipeline
  stamp.cxx)
target_link_libraries(python-vistk.pipeline-stamp
  vistk_pipeline
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(utils
  vistk/pipeline
  utils.cxx)
target_link_libraries(python-vistk.pipeline-utils
  vistk_pipeline
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

make_python_init_file(vistk/pipeline)
#  config
#  datum
#  edge
#  modules
#  pipeline
#  process
#  process_registry
#  schedule
#  schedule_registry
#  stamp)
