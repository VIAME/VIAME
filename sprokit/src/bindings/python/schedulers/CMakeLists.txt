project(sprokit_python_schedulers)

#set(no_install TRUE)

sprokit_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pythread_per_process_scheduler.py
  sprokit/schedulers
  pythread_per_process_scheduler)
