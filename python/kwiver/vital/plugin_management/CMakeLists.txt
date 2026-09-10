set( THIS_MODULE vital/plugin_management )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py "${THIS_MODULE}" __init__ )

################################################################################
# Python Module Library: _interface
kwiver_add_python_library(
  _plugin_management
  "${THIS_MODULE}"
  SOURCES module_plugin_manager.cxx
  PRIVATE vital
)
