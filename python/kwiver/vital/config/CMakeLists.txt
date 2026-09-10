set( THIS_MODULE vital/config )

set( vital_config_python_headers
     module_config_helpers.h
    )

set( vital_config_python_sources
     module_config.cxx
     module_config_helpers.cxx
   )
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py "${THIS_MODULE}" __init__ )

kwiver_add_python_library( _config
      "${THIS_MODULE}"
      SOURCES
      ${vital_config_python_headers}
      ${vital_config_python_sources}
      PRIVATE pybind11::pybind11
              vital_config
              vital
    )
