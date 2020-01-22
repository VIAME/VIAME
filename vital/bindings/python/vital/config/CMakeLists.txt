include_directories(${pybind11_INCLUDE_DIR})

set( vital_config_python_headers
     config.h
    )

set( vital_config_python_sources
     config.cxx
     config_module.cxx
   )

kwiver_add_python_library( config
      vital/config
      ${vital_config_python_headers}
      ${vital_config_python_sources}
    )

target_link_libraries( python-vital.config-config
  PRIVATE              ${PYTHON_LIBRARIES}
                       vital_config
                     )

kwiver_create_python_init( vital/config
  config
)
