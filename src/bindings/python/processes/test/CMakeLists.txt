project(sprokit_python_processes_test)

set(no_install TRUE)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/processes/test
  __init__)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/examples.py
  sprokit/processes/test
  examples)
