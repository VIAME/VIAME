project(sprokit_python_test)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/test.py
  sprokit/test
  test)

kwiver_create_python_init(sprokit/test)

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(python)
endif ()
