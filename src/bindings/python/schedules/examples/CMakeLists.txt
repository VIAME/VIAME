project(vistk_python_schedules_examples)

set(suppress_install TRUE)

add_python_module(__init__.py
  vistk/schedules/examples
  __init__)

add_python_module(pythread_per_process_schedule.py
  vistk/schedules/examples
  pythread_per_process_schedule)
