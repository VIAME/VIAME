project(vital_python_modules)

set(modules_python_srcs
  registration.cxx)

#set(modules_python_private_headers
#  modules-config.h
#  )

set(python_noarch FALSE)
kwiver_create_python_init(vital/modules)
set(python_noarch TRUE)
kwiver_create_python_init(vital/modules)
set(python_noarch FALSE)

#kwiver_private_header_group(${modules_python_private_headers})

if(UNIX)
  add_definitions( -DVITAL_LOAD_PYLIB_SYM )
endif()

if (KWIVER_ENABLE_TOOLS)
kwiver_add_plugin( modules_python
  SUBDIR          ${kwiver_plugin_module_subdir}
  SOURCES         ${modules_python_srcs}
  PRIVATE         vital_python_util
                  kwiversys
                  vital_logger
                  ${${pybind11_library}}
                  ${PYTHON_LIBRARIES}
                  )
 endif()

kwiver_add_python_module("${CMAKE_CURRENT_SOURCE_DIR}/modules.py"
  vital/modules
  modules)

kwiver_add_python_module("${CMAKE_CURRENT_SOURCE_DIR}/loaders.py"
  vital/modules
  loaders)
