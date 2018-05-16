project(vital_python_types)

include_directories(${PYBIND11_INCLUDE_DIR})

kwiver_add_python_library(bounding_box
  vital/types
  bounding_box.cxx)
target_link_libraries(python-vital.types-bounding_box
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(camera
  vital/types
  camera.cxx)
target_link_libraries(python-vital.types-camera
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(camera_intrinsics
  vital/types
  camera_intrinsics.cxx)
target_link_libraries(python-vital.types-camera_intrinsics
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(camera_map
  vital/types
  camera_map.cxx)
target_link_libraries(python-vital.types-camera_map
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(color
  vital/types
  color.cxx)
target_link_libraries(python-vital.types-color
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(covariance
  vital/types
  covariance.cxx)
target_link_libraries(python-vital.types-covariance
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(descriptor
  vital/types
  descriptor.cxx)
target_link_libraries(python-vital.types-descriptor
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(descriptor_set
  vital/types
  descriptor_set.cxx)
target_link_libraries(python-vital.types-descriptor_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(detected_object
  vital/types
  detected_object.cxx)
target_link_libraries(python-vital.types-detected_object
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(detected_object_set
  vital/types
  detected_object_set.cxx)
target_link_libraries(python-vital.types-detected_object_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(detected_object_type
  vital/types
  detected_object_type.cxx)
target_link_libraries(python-vital.types-detected_object_type
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(eigen
  vital/types
  eigen.cxx)
target_link_libraries(python-vital.types-eigen
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(feature
  vital/types
  feature.cxx)
target_link_libraries(python-vital.types-feature
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(homography
  vital/types
  homography.cxx)
target_link_libraries(python-vital.types-homography
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(image
  vital/types
  image.cxx)
target_link_libraries(python-vital.types-image
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(image_container
  vital/types
  image_container.cxx)
target_link_libraries(python-vital.types-image_container
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(landmark
  vital/types
  landmark.cxx)
target_link_libraries(python-vital.types-landmark
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(rotation
  vital/types
  rotation.cxx)
target_link_libraries(python-vital.types-rotation
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(similarity
  vital/types
  similarity.cxx)
target_link_libraries(python-vital.types-similarity
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(timestamp
  vital/types
  timestamp.cxx)
target_link_libraries(python-vital.types-timestamp
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(track
  vital/types
  track.cxx)
target_link_libraries(python-vital.types-track
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(track_set
  vital/types
  track_set.cxx)
target_link_libraries(python-vital.types-track_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

kwiver_add_python_library(object_track_set
  vital/types
  object_track_set.cxx)
target_link_libraries(python-vital.types-object_track_set
  PUBLIC ${PYTHON_LIBRARIES}
         vital)

sprokit_create_python_init(vital/types
  bounding_box
  camera
  camera_intrinsics
  camera_map
  color
  covariance
  descriptor
  descriptor_set
  detected_object_type
  image
  image_container
  detected_object
  detected_object_set
  eigen
  feature
  homography
  landmark
  rotation
  similarity
  timestamp
  track
  track_set
  object_track_set
)
