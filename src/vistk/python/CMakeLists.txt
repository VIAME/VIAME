set(boost_python_library
  python)

if (VISTK_ENABLE_PYTHON3)
  set(boost_python_library
    python3)
endif (VISTK_ENABLE_PYTHON3)

string(TOUPPER ${boost_python_library} boost_python_library_upper)
set(boost_python_library_var
  Boost_${boost_python_library_upper}_LIBRARY)

find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    ${boost_python_library})

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

find_package(PythonInterp EXACT ${PYTHON_VERSION})

find_package(NumPy)

add_subdirectory(util)
add_subdirectory(any_conversion)

if (NUMPY_FOUND)
  add_subdirectory(numpy)
endif (NUMPY_FOUND)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    "${CMAKE_CURRENT_SOURCE_DIR}"
    python)
endif (VISTK_ENABLE_DOCUMENTATION)
