project(sprokit_python_test)

set(no_install TRUE)

add_subdirectory(python)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/test.py
  sprokit/test
  test)

sprokit_create_python_init(sprokit/test)
