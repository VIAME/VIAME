project(vital_python_types)

include_directories(${PYBIND11_INCLUDE_DIR})

sprokit_add_python_library(_bounding_box
  vital/types/bindings
  bounding_box.cxx)
target_link_libraries(python-vital.types.bindings-_bounding_box
  PUBLIC ${PYTHON_LIBRARIES}
         vital_c)

sprokit_add_python_library(_camera
  vital/types/bindings
  camera.cxx)
target_link_libraries(python-vital.types.bindings-_camera
  PUBLIC ${PYTHON_LIBRARIES}
         vital_c)

sprokit_add_python_library(_camera_intrinsics
  vital/types/bindings
  camera_intrinsics.cxx)
target_link_libraries(python-vital.types.bindings-_camera_intrinsics
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_covariance
  vital/types/bindings
  covariance.cxx)
target_link_libraries(python-vital.types.bindings-_covariance
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_descriptor
  vital/types/bindings
  descriptor.cxx)
target_link_libraries(python-vital.types.bindings-_descriptor
  PUBLIC ${PYTHON_LIBRARIES}
         vital_c)

sprokit_add_python_library(_eigen
  vital/types/bindings
  eigen.cxx)
target_link_libraries(python-vital.types.bindings-_eigen
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_rotation
  vital/types/bindings
  rotation.cxx)
target_link_libraries(python-vital.types.bindings-_rotation
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_create_python_init(vital/types/bindings
  _bounding_box
  _camera
  _camera_intrinsics
  _covariance
  _descriptor
  _eigen
  _rotation
)
