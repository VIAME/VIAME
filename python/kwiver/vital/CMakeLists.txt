#
# Python support
#
project(kwiver_python)

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/vital_logging.py
  vital
  vital_logging )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital
  __init__  )

add_subdirectory( util )
add_subdirectory( modules )
add_subdirectory( algo )
add_subdirectory( exceptions )
add_subdirectory( types )

if(KWIVER_ENABLE_TESTS)
  add_subdirectory( tests )
endif()

add_subdirectory( config )
