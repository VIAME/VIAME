project(sprokit_python_processes_test)

set(suppress_install TRUE)

add_python_module(__init__.py
  sprokit/processes/test
  __init__)

add_python_module(examples.py
  sprokit/processes/test
  examples)
