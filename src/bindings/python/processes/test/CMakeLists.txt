project(vistk_python_processes_test)

set(suppress_install TRUE)

add_python_module(__init__.py
  vistk/processes/test
  __init__)

add_python_module(examples.py
  vistk/processes/test
  examples)
