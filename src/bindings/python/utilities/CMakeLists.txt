project(vistk_python_utilities)

add_vistk_python_library(homography
  vistk/utilities
  homography.cxx)
target_link_libraries(python-vistk.utilities-homography
  vistk_utilities
  vistk_python_any_conversion
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

add_vistk_python_library(timestamp
  vistk/utilities
  timestamp.cxx)
target_link_libraries(python-vistk.utilities-timestamp
  vistk_utilities
  vistk_python_any_conversion
  vistk_python_util
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

make_python_init_file(vistk/utilities)
#  homography
#  timestamp)
