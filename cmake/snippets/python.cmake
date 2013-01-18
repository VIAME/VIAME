# Set up options for Python.

option(VISTK_ENABLE_PYTHON "Enable Python bindings" OFF)
cmake_dependent_option(VISTK_ENABLE_PYTHON3 "Use Python3" OFF
  VISTK_ENABLE_PYTHON OFF)
if (VISTK_ENABLE_PYTHON)
  if (WIN32)
    set(destdir bin)
  else ()
    set(destdir lib)
  endif ()

  set(python_output_path "${vistk_binary_dir}/${destdir}/python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")

  set(PYTHON_VERSION "2.7"
    CACHE STRING "The version of python to use for bindings")
  set(PYTHON_ABIFLAGS ""
    CACHE STRING "The ABI flags for the version of Python being used")

  if (VISTK_ENABLE_PYTHON3)
    set(Python_ADDITIONAL_VERSIONS
      3
      ${PYTHON_VERSION})
  endif ()

  # This is to avoid Boost.Python's headers to have __declspec(dllimport) in
  # the headers which confuses Visual Studio's linker.
  cmake_dependent_option(VISTK_HACK_LINK_BOOST_PYTHON_STATIC "Link Boost.Python statically" ON
    WIN32 OFF)
  mark_as_advanced(VISTK_HACK_LINK_BOOST_PYTHON_STATIC)
  if (VISTK_HACK_LINK_BOOST_PYTHON_STATIC)
    add_definitions(-DBOOST_PYTHON_STATIC_LIB)
  endif ()

  if (CMAKE_VERSION VERSION_LESS "2.8.8")
    message(WARNING "Python 3 support may not work with CMake versions older than 2.8.8")
  endif ()
endif ()
