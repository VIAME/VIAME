project(vital_python_util)

include_directories(${pybind11_INCLUDE_DIR})

set(python_util_srcs
  python_exceptions.cxx)

set(python_util_headers
  pybind11.h
  python_exceptions.h
  python.h)

include_directories(${pybind11_INCLUDE_DIR})

kwiver_add_library(vital_python_util
  ${python_util_srcs}
  ${python_util_headers})

target_link_libraries(vital_python_util
  PUBLIC     ${PYTHON_LIBRARIES}
  )

kwiver_install_headers(
    SUBDIR     vital/bindings/python/vital/util
    ${python_util_headers}
)

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/vital_python_util_export.h
  SUBDIR vital/bindings/vital/python/util
  NOPATH
)


kwiver_create_python_init( vital/util )
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/VitalPIL.py
  vital/util
  VitalPIL )
