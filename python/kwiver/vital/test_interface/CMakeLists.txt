set( THIS_MODULE vital/test_interface )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py "${THIS_MODULE}" __init__ )
# TODO: Move this impl to a separate example python package.
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/python_say.py "${THIS_MODULE}" python_say )
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/python_they_say.py "${THIS_MODULE}" python_they_say )

################################################################################
# Python Module Library: _interface
kwiver_add_python_library(
  _interface
  "${THIS_MODULE}"
  SOURCES module_interface.cxx
  PRIVATE pybind11::pybind11
          vital
)
