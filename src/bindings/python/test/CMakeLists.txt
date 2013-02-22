project(vistk_python_test)

set(suppress_install TRUE)

add_subdirectory(python)

add_python_module(test.py
  vistk/test
  test)

make_python_init_file(vistk/test)
