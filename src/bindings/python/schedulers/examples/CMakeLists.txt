project(sprokit_python_schedulers_examples)

set(no_install TRUE)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/schedulers/examples
  __init__)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pythread_per_process_scheduler.py
  sprokit/schedulers/examples
  pythread_per_process_scheduler)
