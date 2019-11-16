project(sprokit_python_sprokit_pipeline)

kwiver_add_python_library(datum
  sprokit/pipeline
  datum.cxx)
target_link_libraries(python-sprokit.pipeline-datum
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital vital_util)

kwiver_add_python_library(edge
  sprokit/pipeline
  edge.cxx)
target_link_libraries(python-sprokit.pipeline-edge
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(pipeline
  sprokit/pipeline
  pipeline.cxx)
target_link_libraries(python-sprokit.pipeline-pipeline
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(process
  sprokit/pipeline
  process.cxx)
target_link_libraries(python-sprokit.pipeline-process
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_util)

kwiver_add_python_library(process_cluster
  sprokit/pipeline
  process_cluster.cxx)
target_link_libraries(python-sprokit.pipeline-process_cluster
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(process_factory
  sprokit/pipeline
  process_factory.cxx)
target_link_libraries(python-sprokit.pipeline-process_factory
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_vpm
    )

kwiver_add_python_library(scheduler
  sprokit/pipeline
  scheduler.cxx)
target_link_libraries(python-sprokit.pipeline-scheduler
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(scheduler_factory
  sprokit/pipeline
  scheduler_factory.cxx)
target_link_libraries(python-sprokit.pipeline-scheduler_factory
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_vpm
  )

kwiver_add_python_library(stamp
  sprokit/pipeline
  stamp.cxx)
target_link_libraries(python-sprokit.pipeline-stamp
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(utils
  sprokit/pipeline
  utils.cxx)
target_link_libraries(python-sprokit.pipeline-utils
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(version
  sprokit/pipeline
  version.cxx)
target_link_libraries(python-sprokit.pipeline-version
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         sprokit/pipeline
                         __init__)

#kwiver_create_python_init(sprokit/pipeline
#  datum
#  edge
#  pipeline
#  process
#  process_factory
#  scheduler
#  scheduler_factory
#  stamp
#  utils
#  version)
