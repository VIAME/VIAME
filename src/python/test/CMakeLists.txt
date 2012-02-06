project(vistk_python_test)

set(no_install_plugin TRUE)

add_vistk_python_library(test_image
  vistk/test
  test_image.cxx)
target_link_libraries(python-vistk.test-test_image
  vil
  ${Boost_PYTHON_LIBRARY}
  ${PYTHON_LIBRARIES})

make_python_init_file(vistk/test)
#  test_image)
