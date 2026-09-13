project(sprokit_python_schedulers)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         sprokit/schedulers
                         __init__)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pythread_per_process_scheduler.py
  sprokit/schedulers
  pythread_per_process)
