project(vital_python_util)

include_directories(${pybind11_INCLUDE_DIR})

set(python_util_srcs
  python_exceptions.cxx)

set(python_util_headers
  pybind11.h
  python_exceptions.h
  python.h)

kwiver_add_library(vital_python_util
  ${python_util_srcs}
  ${python_util_headers})

target_link_libraries(vital_python_util
  PUBLIC     ${PYTHON_LIBRARIES}
  )

kwiver_install_headers(
    SUBDIR     python/vital/util
    ${python_util_headers}
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_python_util_export.h
  SUBDIR python/vital/util
  NOPATH
)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         vital/util
                         __init__)

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/VitalPIL.py
  vital/util
  VitalPIL )

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/find_python_library.py
                         vital/util
                         find_python_library)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/entrypoint.py
                         vital/util
                         entrypoint)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/initial_plugin_path.py
                         vital/util
                         initial_plugin_path)
