project(vistk_python_helpers)

find_package(PythonInterp ${PYTHON_VERSION})

vistk_add_helper_library(python_exceptions
  ${PYTHON_LIBRARIES})

vistk_add_helper_library(pystream
  vistk_python_util
  ${Boost_IOSTREAMS_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${${boost_python_library_var}}
  ${PYTHON_LIBRARIES})

vistk_add_helper_library(python_threading
  ${PYTHON_LIBRARIES})

find_package(NumPy)

if (NUMPY_FOUND)
  include_directories(SYSTEM ${NUMPY_INCLUDE_DIR})

  vistk_add_helper_library(numpy_memory_chunk
    vistk_python_util
    vil
    ${PYTHON_LIBRARIES})
endif ()
