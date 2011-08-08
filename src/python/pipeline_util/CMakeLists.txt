project(vistk_python_pipeline_util)

add_vistk_python_library(bake
  vistk/pipeline_util
  bake.cxx)
target_link_libraries(python-bake
  vistk_pipeline
  vistk_pipeline_util
  pystream
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(export_
  vistk/pipeline_util
  export.cxx)
target_link_libraries(python-export_
  vistk_pipeline
  vistk_pipeline_util
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(load
  vistk/pipeline_util
  load.cxx)
target_link_libraries(python-load
  vistk_pipeline
  vistk_pipeline_util
  pystream
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

make_python_init_file(vistk/pipeline_util
  bake
  export_
  load)
