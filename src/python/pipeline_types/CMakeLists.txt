project(vistk_python_pipeline_types)

add_vistk_python_library(basic
  vistk/pipeline_types
  basic.cxx)
target_link_libraries(python-basic
  vistk_pipeline_types
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(image
  vistk/pipeline_types
  image.cxx)
target_link_libraries(python-image
  vistk_pipeline_types
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(utility
  vistk/pipeline_types
  utility.cxx)
target_link_libraries(python-utility
  vistk_pipeline_types
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

make_python_init_file(vistk/pipeline_types)
#  basic
#  image
#  utility)
