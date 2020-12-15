project(vital_python_modules)

include_directories(${pybind11_INCLUDE_DIR})

set(modules_python_srcs
    module_helpers.h
    module_helpers.cxx
    registration.cxx)

if(UNIX)
  add_definitions( -DVITAL_LOAD_PYLIB_SYM )
endif()

if (KWIVER_ENABLE_TOOLS)
  kwiver_add_plugin( modules_python
    SUBDIR           ${kwiver_plugin_module_subdir}
    SOURCES          ${modules_python_srcs}
    PRIVATE          vital_python_util
                     kwiversys
                     vital_logger
                     vital_vpm
                     ${${pybind11_library}}
                     ${PYTHON_LIBRARIES}
                   )
endif()

kwiver_add_python_module("${CMAKE_CURRENT_SOURCE_DIR}/module_loader.py"
  vital/modules
  module_loader)

kwiver_add_python_module("${CMAKE_CURRENT_SOURCE_DIR}/loaders.py"
  vital/modules
  loaders)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         vital/modules
                         __init__)

kwiver_add_python_library(modules
                          vital/modules
                          modules.cxx)

target_link_libraries(python-vital.modules-modules
    PRIVATE   vital_python_util
              vital_vpm)
