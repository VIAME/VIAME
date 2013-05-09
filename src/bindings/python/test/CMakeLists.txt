project(sprokit_python_test)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/test.py
  sprokit/test
  test)

sprokit_create_python_init(sprokit/test)

if (SPROKIT_ENABLE_TESTING)
  add_subdirectory(python)
endif ()
