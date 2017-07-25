# Set up options for Python.

cmake_dependent_option(SPROKIT_ENABLE_PYTHON3 "Use Python3" OFF
  KWIVER_ENABLE_PYTHON OFF)
if (KWIVER_ENABLE_PYTHON)
  set(sprokit_python_subdir "python${PYTHON_VERSION}${PYTHON_ABIFLAGS}")
  set(sprokit_python_output_path "${KWIVER_BINARY_DIR}/lib")

  set(PYTHON_VERSION "2.7"
    CACHE STRING "The version of python to use for bindings")
  set(PYTHON_ABIFLAGS ""
    CACHE STRING "The ABI flags for the version of Python being used")

  if (SPROKIT_ENABLE_PYTHON3)
    set(Python_ADDITIONAL_VERSIONS
      3
      ${PYTHON_VERSION})

    if (PYTHON_VERSION VERSION_LESS "3.0")
      set(PYTHON_VERSION "3.0"
        CACHE STRING "The version of python to use for bindings" FORCE)
    endif ()
  endif ()

  # This is to avoid Boost.Python's headers to have __declspec(dllimport) in
  # the headers which confuses Visual Studio's linker.
  cmake_dependent_option(SPROKIT_HACK_LINK_BOOST_PYTHON_STATIC "Link Boost.Python statically" ON
    WIN32 OFF)
  mark_as_advanced(SPROKIT_HACK_LINK_BOOST_PYTHON_STATIC)
  if (SPROKIT_HACK_LINK_BOOST_PYTHON_STATIC)
    add_definitions(-DBOOST_PYTHON_STATIC_LIB)
  endif ()
endif ()
