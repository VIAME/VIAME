find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    python)

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

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
