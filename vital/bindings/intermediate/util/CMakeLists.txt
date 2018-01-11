project(vital_python_util)

include_directories(${PYBIND11_INCLUDE_DIR})

sprokit_add_python_library(error_handle
  vital/util/bindings
  error_handle.cxx)
target_link_libraries(python-vital.util.bindings-error_handle
  PUBLIC ${PYTHON_LIBRARIES}
         vital_c)

sprokit_create_python_init(vital/util/bindings
  error_handle
)
