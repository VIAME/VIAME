set( THIS_MODULE vital/io )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py "${THIS_MODULE}" __init__ )

kwiver_add_python_library(
  metadata_io
  "${THIS_MODULE}"
  SOURCES metadata_io.cxx
  PRIVATE pybind11::pybind11
          vital
)

kwiver_add_python_library(
  camera_from_metadata
  "${THIS_MODULE}"
  SOURCES camera_from_metadata.cxx
  PRIVATE pybind11::pybind11
          vital
)
