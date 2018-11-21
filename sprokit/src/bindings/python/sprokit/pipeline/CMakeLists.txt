project(sprokit_python_sprokit_pipeline)

_sprokit_add_python_library(config
  sprokit/pipeline
  config.cxx)
target_link_libraries(python-sprokit.pipeline-config
  PRIVATE             sprokit_pipeline
                      sprokit_python_util
                      vital_exceptions)

_sprokit_add_python_library(datum
  sprokit/pipeline
  datum.cxx)
target_link_libraries(python-sprokit.pipeline-datum
  PRIVATE             sprokit_pipeline
                      sprokit_python_util
                      vital)

_sprokit_add_python_library(edge
  sprokit/pipeline
  edge.cxx)
target_link_libraries(python-sprokit.pipeline-edge
  PRIVATE             sprokit_pipeline
                      sprokit_python_util)

_sprokit_add_python_library(modules
  sprokit/pipeline
  modules.cxx)
target_link_libraries(python-sprokit.pipeline-modules
  PRIVATE             sprokit_pipeline
                      sprokit_python_util
                      ${Boost_SYSTEM_LIBRARY}
                      vital_vpm
  )

_sprokit_add_python_library(pipeline
  sprokit/pipeline
  pipeline.cxx)
target_link_libraries(python-sprokit.pipeline-pipeline
  PRIVATE             sprokit_pipeline
                      sprokit_python_util)

_sprokit_add_python_library(process
  sprokit/pipeline
  process.cxx)
target_link_libraries(python-sprokit.pipeline-process
  PRIVATE             sprokit_pipeline
                      sprokit_python_util
                      vital_util)

_sprokit_add_python_library(process_cluster
  sprokit/pipeline
  process_cluster.cxx)
target_link_libraries(python-sprokit.pipeline-process_cluster
  PRIVATE             sprokit_pipeline
                      sprokit_python_util)

_sprokit_add_python_library(process_factory
  sprokit/pipeline
  process_factory.cxx)
target_link_libraries(python-sprokit.pipeline-process_factory
  PRIVATE             sprokit_pipeline
                      sprokit_python_util
                      vital_vpm
    )

_sprokit_add_python_library(scheduler
  sprokit/pipeline
  scheduler.cxx)
target_link_libraries(python-sprokit.pipeline-scheduler
  PRIVATE             sprokit_pipeline
                      sprokit_python_util)

_sprokit_add_python_library(scheduler_factory
  sprokit/pipeline
  scheduler_factory.cxx)
target_link_libraries(python-sprokit.pipeline-scheduler_factory
  PRIVATE             sprokit_pipeline
                      sprokit_python_util
                      vital_vpm
  )

_sprokit_add_python_library(stamp
  sprokit/pipeline
  stamp.cxx)
target_link_libraries(python-sprokit.pipeline-stamp
  PRIVATE             sprokit_pipeline
                      sprokit_python_util)

_sprokit_add_python_library(utils
  sprokit/pipeline
  utils.cxx)
target_link_libraries(python-sprokit.pipeline-utils
  PRIVATE             sprokit_pipeline
                      sprokit_python_util)

_sprokit_add_python_library(version
  sprokit/pipeline
  version.cxx)
target_link_libraries(python-sprokit.pipeline-version
  PRIVATE             sprokit_pipeline
                      sprokit_python_util)

kwiver_create_python_init(sprokit/pipeline
  config
  datum
  edge
  modules
  pipeline
  process
  process_factory
  scheduler
  scheduler_factory
  stamp
  utils
  version)
