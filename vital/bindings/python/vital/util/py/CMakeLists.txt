#
# Python support
#

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/array_wrapping.py
  vital/util
  array_wrapping )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/find_vital_library.py
  vital/util
  find_vital_library )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/string.py
  vital/util
  string )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/VitalObject.py
  vital/util
  VitalObject )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/VitalPIL.py
  vital/util
  VitalPIL )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital/util
  __init__ )
