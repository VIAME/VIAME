project(vital_python_modules)

include_directories(${pybind11_INCLUDE_DIR})
set(modules_python_srcs
  registration.cxx)

set(python_noarch FALSE)
kwiver_create_python_init(vital/modules)
set(python_noarch TRUE)
kwiver_create_python_init(vital/modules)
set(python_noarch FALSE)


if(UNIX)
  add_definitions( -DVITAL_LOAD_PYLIB_SYM )
endif()

if (KWIVER_ENABLE_TOOLS)
kwiver_add_plugin( modules_python
  SUBDIR          ${kwiver_plugin_process_subdir}
  SOURCES         ${modules_python_srcs}
  PRIVATE         vital_python_util
                  kwiversys
                  vital_logger
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

kwiver_add_python_library(modules
                          vital/modules
                          modules.cxx)

target_link_libraries(python-vital.modules-modules
    PRIVATE   vital_python_util
              vital_vpm)
