#
# Python supporot
#

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.py
  vital/algo
  algorithm )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/bundle_adjust.py
  vital/algo
  bundle_adjust )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/convert_image.py
  vital/algo
  convert_image )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/estimate_canonical_transform.py
  vital/algo
  estimate_canonical_transform )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/estimate_similarity_transform.py
  vital/algo
  estimate_similarity_transform )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/geo_map.py
  vital/algo
  geo_map )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/image_io.py
  vital/algo
  image_io )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/initialize_cameras_landmarks.py
  vital/algo
  initialize_cameras_landmarks )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track_features.py
  vital/algo
  track_features )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/triangulate_landmarks.py
  vital/algo
  triangulate_landmarks )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital/algo
  __init__ )

#kwiver_create_python_init( vital/algo
#  algorithm
#  convert_image
#  image
#  track_features
#  )
