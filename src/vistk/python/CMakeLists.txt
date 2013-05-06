set(boost_python_library
  python)

if (SPROKIT_ENABLE_PYTHON3)
  set(boost_python_library
    python3)
endif ()

string(TOUPPER ${boost_python_library} boost_python_library_upper)
set(boost_python_library_var
  Boost_${boost_python_library_upper}_LIBRARY)

find_package(Boost ${sprokit_boost_version} REQUIRED
  COMPONENTS
    ${boost_python_library})

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_DIR})

find_package(PythonInterp ${PYTHON_VERSION})

add_subdirectory(util)
add_subdirectory(any_conversion)

create_doxygen(
  "${CMAKE_CURRENT_SOURCE_DIR}"
  python)
