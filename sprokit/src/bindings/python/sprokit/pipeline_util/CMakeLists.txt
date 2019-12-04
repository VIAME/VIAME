project(sprokit_python_sprokit_pipeline_util)

if( NOT WIN32 )

kwiver_add_python_library(bake
  sprokit/pipeline_util
  bake.cxx)

target_link_libraries(python-sprokit.pipeline_util-bake
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    sprokit_python_util
    vital_python_util
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    vital_vpm
  )

kwiver_add_python_library(export_
  sprokit/pipeline_util
  export.cxx)

target_link_libraries(python-sprokit.pipeline_util-export_
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    vital_python_util
    sprokit_python_util)


kwiver_add_python_library(load
  sprokit/pipeline_util
  load.cxx)
target_link_libraries(python-sprokit.pipeline_util-load
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    sprokit_python_util
    vital_python_util
    vital_util
    ${Boost_IOSTREAMS_LIBRARY})

kwiver_create_python_init(sprokit/pipeline_util
  bake
  export_
  load)

endif()
