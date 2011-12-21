find_package(Boost ${vistk_boost_version} REQUIRED
  COMPONENTS
    python)

find_package(PythonLibs REQUIRED)
include_directories(SYSTEM ${PYTHON_INCLUDE_PATH})

add_subdirectory(any_conversion)
