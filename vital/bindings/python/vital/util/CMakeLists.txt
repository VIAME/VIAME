project(vital_python_util)


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


add_subdirectory(py)
