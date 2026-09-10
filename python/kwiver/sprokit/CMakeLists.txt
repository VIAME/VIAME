project(sprokit_python_sprokit)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         sprokit
                         __init__)

add_subdirectory(pipeline)
add_subdirectory(pipeline_util)
add_subdirectory(schedulers)
add_subdirectory(processes)
add_subdirectory(adapters)

if(NOT WIN32)
  add_subdirectory(util)
endif()

if(KWIVER_ENABLE_PYTHON_TESTS AND NOT SKBUILD)
  add_subdirectory(tests)
endif()
