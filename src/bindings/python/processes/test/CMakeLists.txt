project(vistk_python_processes_test)

set(suppress_install TRUE)

add_python_module(examples.py
  vistk/processes/test
  examples)

make_python_init_file(vistk/processes/test)
#  examples)
