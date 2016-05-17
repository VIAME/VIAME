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
else()
  string(STRIP "${python_site_packages}" python_site_packages)
  message(STATUS "Python site-packages to install into: ${python_site_packages}")
endif()

# Install all python files
kwiver_install( DIRECTORY      kwiver
  DESTINATION                  ${python_site_packages}
  )
