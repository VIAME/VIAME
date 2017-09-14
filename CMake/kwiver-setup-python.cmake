###
# Finds the python binaries, libraries, inclue, and site-packages paths


set(KWIVER_PYTHON_VERSION "2" CACHE STRING "Python version to use: 3 or 2")
set_property(CACHE KWIVER_PYTHON_VERSION PROPERTY STRINGS "3" "2")

# If we change python versions re-find the bin, include, and libs
if (NOT _prev_kwiver_python_version STREQUAL KWIVER_PYTHON_VERSION)
  # but dont clobber initial settings in the instance they are specified via
  # commandline (e.g  cmake -DPYTHON_EXECUTABLE=/my/special/python)
  if (_prev_kwiver_python_version)
    message(STATUS "The Python version changed; refinding the interpreter")
    unset(PYTHON_EXECUTABLE CACHE)
    unset(PYTHON_INCLUDE_DIR CACHE)
    unset(PYTHON_LIBRARY CACHE)
    unset(PYTHON_LIBRARY_DEBUG CACHE)
  endif()
endif()

if (KWIVER_PYTHON_VERSION MATCHES "^3.*")
  find_package(PythonInterp 3.4 REQUIRED)
  find_package(PythonLibs 3.4 REQUIRED)
else()
  find_package(PythonInterp 2.7 REQUIRED)
  find_package(PythonLibs 2.7 REQUIRED)
endif()

# Make a copy so we can determine if the user changes python versions
set(_prev_kwiver_python_version "${KWIVER_PYTHON_VERSION}" CACHE INTERNAL
  "allows us to determine if the user changes python version")


###
#
# Get canonical directory for python site packages.
# It varys from system to system.
#
execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import distutils.sysconfig; print(distutils.sysconfig.get_python_lib(prefix=''))"
  RESULT_VARIABLE proc_success
  OUTPUT_VARIABLE python_site_packages
  )

if(NOT ${proc_success} EQUAL 0)
  message(FATAL_ERROR "Request for python site-packages location failed with error code: ${proc_success}")
endif()

# Current usage determines most of the path in alternate ways.
# All we need to supply is the '*-packages' directory name.
# Customers could be converted to accept a larger part of the path from this function.
string( REGEX MATCH "dist-packages" result ${python_site_packages} )
if (result)
  set( python_sitename "dist-packages")
else()
  set( python_sitename "site-packages")
endif()
string(STRIP "${python_site_packages}" python_site_packages)
message(STATUS "Python site-packages to install into: ${python_site_packages}")
message(STATUS "python_site_packages = ${python_site_packages}")
message(STATUS "python_sitename = ${python_sitename}")


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; print(sys.version[0:3])"
  RESULT_VARIABLE proc_success
  OUTPUT_VARIABLE _python_version
  )
if(NOT ${proc_success} EQUAL 0)
  message(FATAL_ERROR "Request for python version failed with error code: ${proc_success}")
endif()


set(PYTHON_VERSION ${_python_version}
  CACHE STRING "The version of python to use for bindings")
set(PYTHON_ABIFLAGS ""
  CACHE STRING "The ABI flags for the version of Python being used")

include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})


# This is to avoid Boost.Python's headers to have __declspec(dllimport) in
# the headers which confuses Visual Studio's linker.
cmake_dependent_option(SPROKIT_HACK_LINK_BOOST_PYTHON_STATIC "Link Boost.Python statically" ON
  WIN32 OFF)
mark_as_advanced(SPROKIT_HACK_LINK_BOOST_PYTHON_STATIC)
if (SPROKIT_HACK_LINK_BOOST_PYTHON_STATIC)
  add_definitions(-DBOOST_PYTHON_STATIC_LIB)
endif ()


set(kwiver_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
set(kwiver_python_output_path "${KWIVER_BINARY_DIR}/lib/${kwiver_python_subdir}")

set(sprokit_python_output_path "${KWIVER_BINARY_DIR}/lib")

set(PYTHON_CONFIG_STATUS "

PYTHON_CONFIG_STATUS

  * PYTHON_ABIFLAGS = ${PYTHON_ABIFLAGS}
  * PYTHON_VERSION = ${PYTHON_VERSION}

  * KWIVER_PYTHON_VERSION = ${KWIVER_PYTHON_VERSION}

  * PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}
  * PYTHON_INCLUDE_DIR = ${PYTHON_INCLUDE_DIR}
  * PYTHON_LIBRARY = ${PYTHON_LIBRARY}
  * PYTHON_LIBRARY_DEBUG = ${PYTHON_LIBRARY_DEBUG}

  * kwiver_python_subdir = ${kwiver_python_subdir}
  * kwiver_python_output_path = ${kwiver_python_output_path}
  * sprokit_python_output_path = ${sprokit_python_output_path}

  * python_site_packages = ${python_site_packages}
  * python_sitename = ${python_sitename}

")
