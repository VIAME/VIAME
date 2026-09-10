project(sprokit_python_sprokit_pipeline)

kwiver_add_python_library(datum
  sprokit/pipeline
  SOURCES datum.cxx)
target_link_libraries(python-sprokit.pipeline-datum
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital vital_util)

kwiver_add_python_library(edge
  sprokit/pipeline
  SOURCES edge.cxx)
target_link_libraries(python-sprokit.pipeline-edge
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(pipeline
  sprokit/pipeline
  SOURCES pipeline.cxx)
target_link_libraries(python-sprokit.pipeline-pipeline
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(process
  sprokit/pipeline
  SOURCES process.cxx)
target_link_libraries(python-sprokit.pipeline-process
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_util)

kwiver_add_python_library(process_cluster
  sprokit/pipeline
  SOURCES process_cluster.cxx)
target_link_libraries(python-sprokit.pipeline-process_cluster
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(process_factory
  sprokit/pipeline
  SOURCES process_factory.cxx)
target_link_libraries(python-sprokit.pipeline-process_factory
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_vpm
    )

kwiver_add_python_library(scheduler
  sprokit/pipeline
  SOURCES scheduler.cxx)
target_link_libraries(python-sprokit.pipeline-scheduler
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(scheduler_factory
  sprokit/pipeline
  SOURCES scheduler_factory.cxx)
target_link_libraries(python-sprokit.pipeline-scheduler_factory
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_vpm
  )

kwiver_add_python_library(stamp
  sprokit/pipeline
  SOURCES stamp.cxx)
target_link_libraries(python-sprokit.pipeline-stamp
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(utils
  sprokit/pipeline
  SOURCES utils.cxx)
target_link_libraries(python-sprokit.pipeline-utils
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_library(version
  sprokit/pipeline
  SOURCES version.cxx)
target_link_libraries(python-sprokit.pipeline-version
  PRIVATE             sprokit_pipeline
                      vital_python_util)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/pipeline
  __init__)
