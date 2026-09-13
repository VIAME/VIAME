viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  sprokit/tests/processes
  __init__)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/examples.py
  sprokit/tests/processes
  examples)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/extras.py
  sprokit/tests/processes
  extras)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pythonpath_test_process.py
  sprokit/tests/processes
  pythonpath_test_process)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/pythonpath_test_scheduler.py
  sprokit/tests/processes
  pythonpath_test_scheduler)
