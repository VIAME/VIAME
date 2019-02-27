project(vital_python_util)

include_directories(${PYBIND11_INCLUDE_DIR})

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
    SUBDIR     vital/util
    ${python_util_headers}
)


add_subdirectory(py)
