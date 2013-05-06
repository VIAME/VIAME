project(sprokit_python_sprokit_pipeline_util)

add_sprokit_python_library(bake
  sprokit/pipeline_util
  bake.cxx)
target_link_libraries(python-sprokit.pipeline_util-bake
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    sprokit_python_util
    pystream
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

add_sprokit_python_library(export_
  sprokit/pipeline_util
  export.cxx)
target_link_libraries(python-sprokit.pipeline_util-export_
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    sprokit_python_util
    pystream
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

add_sprokit_python_library(load
  sprokit/pipeline_util
  load.cxx)
target_link_libraries(python-sprokit.pipeline_util-load
  LINK_PRIVATE
    sprokit_pipeline
    sprokit_pipeline_util
    sprokit_python_util
    pystream
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

make_python_init_file(sprokit/pipeline_util)
#  bake
#  export_
#  load)
