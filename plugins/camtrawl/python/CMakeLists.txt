# If we make a non-standalone module use this
#kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
#  viame/processes/camtrawl
#  __init__)

# Just install module as a standalone file to the pythonpath
kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/camtrawl_processes.py
  viame/processes camtrawl_processes)
