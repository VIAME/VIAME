project(vital_python_types)

include_directories(${PYBIND11_INCLUDE_DIR})

sprokit_add_python_library(_bounding_box
  vital/types/bindings
  bounding_box.cxx)
target_link_libraries(python-vital.types.bindings-_bounding_box
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_camera
  vital/types/bindings
  camera.cxx)
target_link_libraries(python-vital.types.bindings-_camera
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_camera_intrinsics
  vital/types/bindings
  camera_intrinsics.cxx)
target_link_libraries(python-vital.types.bindings-_camera_intrinsics
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_camera_map
  vital/types/bindings
  camera_map.cxx)
target_link_libraries(python-vital.types.bindings-_camera_map
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_color
  vital/types/bindings
  color.cxx)
target_link_libraries(python-vital.types.bindings-_color
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
         vital)

sprokit_add_python_library(_descriptor_set
  vital/types/bindings
  descriptor_set.cxx)
target_link_libraries(python-vital.types.bindings-_descriptor_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_detected_object
  vital/types/bindings
  detected_object.cxx)
target_link_libraries(python-vital.types.bindings-_detected_object
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_detected_object_set
  vital/types/bindings
  detected_object_set.cxx)
target_link_libraries(python-vital.types.bindings-_detected_object_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_detected_object_type
  vital/types/bindings
  detected_object_type.cxx)
target_link_libraries(python-vital.types.bindings-_detected_object_type
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_eigen
  vital/types/bindings
  eigen.cxx)
target_link_libraries(python-vital.types.bindings-_eigen
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_feature
  vital/types/bindings
  feature.cxx)
target_link_libraries(python-vital.types.bindings-_feature
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_homography
  vital/types/bindings
  homography.cxx)
target_link_libraries(python-vital.types.bindings-_homography
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_image
  vital/types/bindings
  image.cxx)
target_link_libraries(python-vital.types.bindings-_image
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_image_container
  vital/types/bindings
  image_container.cxx)
target_link_libraries(python-vital.types.bindings-_image_container
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_landmark
  vital/types/bindings
  landmark.cxx)
target_link_libraries(python-vital.types.bindings-_landmark
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_rotation
  vital/types/bindings
  rotation.cxx)
target_link_libraries(python-vital.types.bindings-_rotation
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_similarity
  vital/types/bindings
  similarity.cxx)
target_link_libraries(python-vital.types.bindings-_similarity
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_track
  vital/types/bindings
  track.cxx)
target_link_libraries(python-vital.types.bindings-_track
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_track_set
  vital/types/bindings
  track_set.cxx)
target_link_libraries(python-vital.types.bindings-_track_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_add_python_library(_object_track_set
  vital/types/bindings
  object_track_set.cxx)
target_link_libraries(python-vital.types.bindings-_object_track_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_create_python_init(vital/types/bindings
  _bounding_box
  _camera
  _camera_intrinsics
  _camera_map
  _color
  _covariance
  _descriptor
  _descriptor_set
  _detected_object_type
  _detected_object
  _detected_object_set
  _eigen
  _feature
  _homography
  _image
  _image_container
  _landmark
  _rotation
  _similarity
  _track
  _track_set
  _object_track_set
)
