project(vistk_python_schedules_examples)

set(suppress_install TRUE)

add_python_module(pythread_per_process_schedule.py
  vistk/schedules/examples
  pythread_per_process_schedule)

make_python_init_file(vistk/schedules/examples)
#  pythread_per_process_schedule)
