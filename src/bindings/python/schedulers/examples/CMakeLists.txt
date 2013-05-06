project(sprokit_python_schedulers_examples)

set(suppress_install TRUE)

add_python_module(__init__.py
  sprokit/schedulers/examples
  __init__)

add_python_module(pythread_per_process_scheduler.py
  sprokit/schedulers/examples
  pythread_per_process_scheduler)
