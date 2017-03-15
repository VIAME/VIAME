project(sprokit_python_sprokit_pipeline_util)

_sprokit_add_python_library(bake
  sprokit/pipeline_util
  bake.cxx)
target_link_libraries(python-sprokit.pipeline_util-bake
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    sprokit_python_util
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES}
    vital_vpm
  )

_sprokit_add_python_library(export_
  sprokit/pipeline_util
  export.cxx)
target_link_libraries(python-sprokit.pipeline_util-export_
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    sprokit_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

_sprokit_add_python_library(load
  sprokit/pipeline_util
  load.cxx)
target_link_libraries(python-sprokit.pipeline_util-load
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    sprokit_python_util
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

sprokit_create_python_init(sprokit/pipeline_util)
#  bake
#  export_
#  load)
