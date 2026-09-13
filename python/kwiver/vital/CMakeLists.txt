#
# Python support
#
project(kwiver_python)

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/vital_logging.py
  vital
  vital_logging )

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital
  __init__  )

add_subdirectory( util )
add_subdirectory( modules )
# `algo` moved to `library/algorithm_framework/algo` in P8-T02, and the
# castxml generator that wrote its bindings was deleted with it.
add_subdirectory( applets )
add_subdirectory( exceptions )

if(KWIVER_ENABLE_PYTHON_TESTS)
  add_subdirectory( tests )
endif()

add_subdirectory( config )
add_subdirectory( io )
add_subdirectory( plugin_management )
add_subdirectory( plugins )
add_subdirectory( test_interface )
# `types` moved to `library/core_types` in P8-T01; its bindings are built
# by `library/core_types/python.cmake` and keep the module path they had.
