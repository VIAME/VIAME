set( THIS_MODULE vital/plugins )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py "${THIS_MODULE}" __init__ )
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/discovery.py "${THIS_MODULE}" discovery )

################################################################################
# Python Module Library: _pluggable
kwiver_add_python_library(
  _pluggable
  "${THIS_MODULE}"
  SOURCES module_pluggable.cxx
  PRIVATE vital
)
################################################################################
# Plugin library
#
# This library provides the hooks into Python interpreter land to register a
# number of factories for python implementations discovered.
#

set(libraries_to_link pybind11::embed pybind11::pybind11)
if((NOT SKBUILD) OR MSVC)
  # libraries included in a wheel should not link to Python::Python directly
  # When used from python, libpython is already loaded so the plugin is able to find the required symbols.
  set(libraries_to_link "${libraries_to_link}" Python::Python)
endif()

kwiver_add_plugin(
  plugins_from_python
  SOURCES   register.cxx
  PRIVATE   ${libraries_to_link}
            vital
  SUBDIR    ${kwiver_plugin_module_subdir}
)
