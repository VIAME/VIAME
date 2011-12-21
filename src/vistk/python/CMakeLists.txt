find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    python)

find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_PATH})

add_subdirectory(any_conversion)

if (VISTK_ENABLE_DOCUMENTATION)
  create_doxygen(
    ${CMAKE_CURRENT_SOURCE_DIR}
    python)
endif (VISTK_ENABLE_DOCUMENTATION)
