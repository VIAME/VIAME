#
# Python support
#

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/bounding_box.py
  vital/types
  bounding_box )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/camera.py
  vital/types
  camera )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/camera_intrinsics.py
  vital/types
  camera_intrinsics )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/camera_map.py
  vital/types
  camera_map )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/color.py
  vital/types
  color )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/covariance.py
  vital/types
  covariance )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/descriptor.py
  vital/types
  descriptor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/descriptor_set.py
  vital/types
  descriptor_set )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/detected_object.py
  vital/types
  detected_object )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/detected_object_type.py
  vital/types
  detected_object_type )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/detected_object_set.py
  vital/types
  detected_object_set )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/eigen.py
  vital/types
  eigen )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/feature.py
  vital/types
  feature )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/feature_track_set.py
  vital/types
  feature_track_set )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/homography.py
  vital/types
  homography )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/image.py
  vital/types
  image )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/image_container.py
  vital/types
  image_container )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/landmark.py
  vital/types
  landmark )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/landmark_map.py
  vital/types
  landmark_map )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/object_track_set.py
  vital/types
  object_track_set )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/rotation.py
  vital/types
  rotation )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/similarity.py
  vital/types
  similarity )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track.py
  vital/types
  track )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track_set.py
  vital/types
  track_set )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital/types
  __init__ )
