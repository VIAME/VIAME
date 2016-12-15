#
# Python supporot
#

## kwiver_create_python_plugin_init(vital/algo)

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/algorithm.py
  vital/algo
  algorithm )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/convert_image.py
  vital/algo
  convert_image )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/image_io.py
  vital/algo
  image_io )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/track_features.py
  vital/algo
  track_features )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital/algo
  __init__ )

#kwiver_create_python_init( vital/algo
#  algorithm
#  convert_image
#  image
#  track_features
#  )
