project(sprokit_python_sprokit_pipeline)

_sprokit_add_python_library(config
  sprokit/pipeline
  config.cxx)
target_link_libraries(python-sprokit.pipeline-config
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(datum
  sprokit/pipeline
  datum.cxx)
target_link_libraries(python-sprokit.pipeline-datum
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_any_conversion
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(edge
  sprokit/pipeline
  edge.cxx)
target_link_libraries(python-sprokit.pipeline-edge
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(modules
  sprokit/pipeline
  modules.cxx)
target_link_libraries(python-sprokit.pipeline-modules
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(pipeline
  sprokit/pipeline
  pipeline.cxx)
target_link_libraries(python-sprokit.pipeline-pipeline
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(process
  sprokit/pipeline
  process.cxx)
target_link_libraries(python-sprokit.pipeline-process
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    python_exceptions
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(process_cluster
  sprokit/pipeline
  process_cluster.cxx)
target_link_libraries(python-sprokit.pipeline-process_cluster
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    python_exceptions
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(process_registry
  sprokit/pipeline
  process_registry.cxx)
target_link_libraries(python-sprokit.pipeline-process_registry
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    python_exceptions
    python_threading
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(scheduler
  sprokit/pipeline
  scheduler.cxx)
target_link_libraries(python-sprokit.pipeline-scheduler
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    python_exceptions
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(scheduler_registry
  sprokit/pipeline
  scheduler_registry.cxx)
target_link_libraries(python-sprokit.pipeline-scheduler_registry
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    python_threading
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(stamp
  sprokit/pipeline
  stamp.cxx)
target_link_libraries(python-sprokit.pipeline-stamp
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(utils
  sprokit/pipeline
  utils.cxx)
target_link_libraries(python-sprokit.pipeline-utils
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(version
  sprokit/pipeline
  version.cxx)
target_link_libraries(python-sprokit.pipeline-version
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

sprokit_create_python_init(sprokit/pipeline)
#  config
#  datum
#  edge
#  modules
#  pipeline
#  process
#  process_registry
#  scheduler
#  scheduler_registry
#  stamp
#  utils
#  version)
