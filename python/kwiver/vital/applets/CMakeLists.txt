#
# Python applets bindings
#

set( THIS_MODULE vital/applets )

set( python_applets_headers
  kwiver_applet_trampoline.txx
  )

set( python_applets_sources
  kwiver_applet.cxx
  module_applets.cxx
  )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py "${THIS_MODULE}" __init__ )
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/applet_utils.py "${THIS_MODULE}" applet_utils )

kwiver_add_python_library( _applets
  "${THIS_MODULE}"
  SOURCES
  ${python_applets_headers}
  ${python_applets_sources}
  PRIVATE pybind11::pybind11
          vital_applets
          vital_config
          vital
  )
