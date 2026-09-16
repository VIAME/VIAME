project(viame_python_util)

set(python_util_srcs
  python_exceptions.cxx)

set(python_util_headers
  python_exceptions.h
  python.h)

viame_add_library(viame_python_util
  ${python_util_srcs}
  ${python_util_headers})

target_link_libraries(viame_python_util
  PRIVATE $<BUILD_INTERFACE:pybind11::embed> $<BUILD_INTERFACE:pybind11::module>
  )

viame_install_headers(
    SUBDIR     python/vital/util
    ${python_util_headers}
)

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_python_util_export.h
  SUBDIR python/vital/util
  NOPATH
)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                         vital/util
                         __init__)

viame_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/VitalPIL.py
  vital/util
  VitalPIL )

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/find_python_library.py
                         vital/util
                         find_python_library)

viame_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/entrypoint.py
                         vital/util
                         entrypoint)

