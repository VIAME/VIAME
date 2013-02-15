project(vistk_python_test)

set(suppress_install TRUE)

add_subdirectory(python)

add_vistk_python_library(test_image
  vistk/test
  test_image.cxx)
target_link_libraries(python-vistk.test-test_image
  LINK_PRIVATE
    vil
    vistk_python_util
    ${${boost_python_library_var}}
    ${PYTHON_LIBRARIES})

add_python_module(test.py
  vistk/test
  test)

make_python_init_file(vistk/test)
#  test_image)
