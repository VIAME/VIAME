project(vistk_python_processes_test)

set(suppress_install TRUE)

add_python_module(examples.py
  vistk/processes/test
  examples)

add_python_module(extras.py
  vistk/test/python/modules
  extras)

make_python_init_file(vistk/processes/test)
#  examples)
make_python_init_file(vistk/test/python)
make_python_init_file(vistk/test/python/modules)
