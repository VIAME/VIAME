project(vistk_python_vistk_scoring)

add_vistk_python_library(scoring_result
  vistk/scoring
  scoring_result.cxx)
target_link_libraries(python-vistk.scoring-scoring_result
  vistk_scoring
  vistk_python_any_conversion
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

make_python_init_file(vistk/scoring)
#  scoring_result)
