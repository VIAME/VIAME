#
# Python support
#
project(kwiver_python)

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/apm.py
  vital
  apm )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/vital_logging.py
  vital
  vital_logging )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital
  __init__  )

add_subdirectory( util )
add_subdirectory( modules )
add_subdirectory( algo )
add_subdirectory( bin ) ##+ do we really need this (maptk application)
add_subdirectory( exceptions )
add_subdirectory( types )
add_subdirectory( tests )
add_subdirectory( config )
