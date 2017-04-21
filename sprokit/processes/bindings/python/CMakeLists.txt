#
# Kwiver python bindings
#

find_package(PythonLibs)
find_package(PythonInterp REQUIRED)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import distutils.sysconfig; print distutils.sysconfig.get_python_lib(prefix='')"
  RESULT_VARIABLE proc_success
  OUTPUT_VARIABLE python_site_packages
  )

if(NOT ${proc_success} EQUAL 0)
  message(FATAL_ERROR "Request for python site-packages location failed with error code: ${proc_success}")
endif()

# Move down to subdirectories
set(kwiver_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/lib/${kwiver_python_subdir}")

#kwiver_create_python_plugin_init(kwiver)
kwiver_create_python_plugin_init(kwiver/util)

add_subdirectory( kwiver )
