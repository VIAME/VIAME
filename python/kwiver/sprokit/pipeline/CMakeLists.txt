project(sprokit_python_sprokit_pipeline)

viame_add_python_library(datum
  sprokit/pipeline
  SOURCES datum.cxx)
viame_target_link_libraries(python-sprokit.pipeline-datum
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital vital_util)

viame_add_python_library(edge
  sprokit/pipeline
  SOURCES edge.cxx)
viame_target_link_libraries(python-sprokit.pipeline-edge
  PRIVATE             sprokit_pipeline
                      vital_python_util)

viame_add_python_library(pipeline
  sprokit/pipeline
  SOURCES pipeline.cxx)
viame_target_link_libraries(python-sprokit.pipeline-pipeline
  PRIVATE             sprokit_pipeline
                      vital_python_util)

viame_add_python_library(process
  sprokit/pipeline
  SOURCES process.cxx)
viame_target_link_libraries(python-sprokit.pipeline-process
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_util)

viame_add_python_library(process_factory
  sprokit/pipeline
  SOURCES process_factory.cxx)
viame_target_link_libraries(python-sprokit.pipeline-process_factory
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_vpm
    )

viame_add_python_library(scheduler
  sprokit/pipeline
  SOURCES scheduler.cxx)
viame_target_link_libraries(python-sprokit.pipeline-scheduler
  PRIVATE             sprokit_pipeline
                      vital_python_util)

viame_add_python_library(scheduler_factory
  sprokit/pipeline
  SOURCES scheduler_factory.cxx)
viame_target_link_libraries(python-sprokit.pipeline-scheduler_factory
  PRIVATE             sprokit_pipeline
                      vital_python_util
                      vital_vpm
  )

viame_add_python_library(stamp
  sprokit/pipeline
  SOURCES stamp.cxx)
viame_target_link_libraries(python-sprokit.pipeline-stamp
  PRIVATE             sprokit_pipeline
                      vital_python_util)

viame_add_python_library(utils
  sprokit/pipeline
  SOURCES utils.cxx)
viame_target_link_libraries(python-sprokit.pipeline-utils
  PRIVATE             sprokit_pipeline
                      vital_python_util)

viame_add_python_library(version
  sprokit/pipeline
  SOURCES version.cxx)
viame_target_link_libraries(python-sprokit.pipeline-version
  PRIVATE             sprokit_pipeline
                      vital_python_util)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/pipeline
  __init__)
