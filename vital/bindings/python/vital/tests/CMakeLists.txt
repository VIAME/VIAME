#
# Python support
#

## kwiver_create_python_plugin_init(vital/tests)

# Add python modules
kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_algo_convert_image.py
  vital/tests
  test_algo_convert_image )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_camera_intrinsics.py
  vital/tests
  test_camera_intrinsics )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_camera_map.py
  vital/tests
  test_camera_map )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_camera.py
  vital/tests
  test_camera )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_color.py
  vital/tests
  test_color )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_config_block.py
  vital/tests
  test_config_block )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_covariance.py
  vital/tests
  test_covariance )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_descriptor.py
  vital/tests
  test_descriptor )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_eigen_numpy.py
  vital/tests
  test_eigen_numpy )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_feature.py
  vital/tests
  test_feature )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_find_vital_library.py
  vital/tests
  test_find_vital_library )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_image_container.py
  vital/tests
  test_image_container )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_image.py
  vital/tests
  test_image )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_rotation.py
  vital/tests
  test_rotation )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_track.py
  vital/tests
  test_track )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/test_track_set.py
  vital/tests
  test_track_set )

kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  vital/tests
  __init__ )
