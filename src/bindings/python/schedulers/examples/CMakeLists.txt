project(vistk_python_schedulers_examples)

set(suppress_install TRUE)

add_python_module(__init__.py
  vistk/schedulers/examples
  __init__)

add_python_module(pythread_per_process_scheduler.py
  vistk/schedulers/examples
  pythread_per_process_scheduler)
