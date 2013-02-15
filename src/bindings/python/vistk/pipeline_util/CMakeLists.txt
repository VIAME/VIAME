project(vistk_python_vistk_pipeline_util)

add_vistk_python_library(bake
  vistk/pipeline_util
  bake.cxx)
target_link_libraries(python-vistk.pipeline_util-bake
  LINK_PRIVATE
    vistk_pipeline
    vistk_pipeline_util
    vistk_python_util
    pystream
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

add_vistk_python_library(export_
  vistk/pipeline_util
  export.cxx)
target_link_libraries(python-vistk.pipeline_util-export_
  LINK_PRIVATE
    vistk_pipeline
    vistk_pipeline_util
    vistk_python_util
    pystream
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

add_vistk_python_library(load
  vistk/pipeline_util
  load.cxx)
target_link_libraries(python-vistk.pipeline_util-load
  LINK_PRIVATE
    vistk_pipeline
    vistk_pipeline_util
    vistk_python_util
    pystream
    ${Boost_IOSTREAMS_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

make_python_init_file(vistk/pipeline_util)
#  bake
#  export_
#  load)
