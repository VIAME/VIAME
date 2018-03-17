#
# Python support
#

# Define modules with tests in them
set(__testable_modnames
  test_algo_convert_image
  test_camera
  test_camera_intrinsics
  test_camera_map
  test_color
  test_covariance
  test_descriptor
  test_descriptor_set
  test_eigen_numpy
  test_feature
  test_find_vital_library
  test_homography
  test_image
  test_image_container
  test_landmark
  test_rotation
  test_similarity
  test_timestamp
  test_track
  test_track_set
  test_track_state
  )

# All python filenames_we in the vital.tests module
set(__all_modnames
  __init__
  helpers
  ${__testable_modnames}
  )

# Add files to the vital.tests python module
foreach (modname IN LISTS __all_modnames)
  kwiver_add_python_module( ${CMAKE_CURRENT_SOURCE_DIR}/${modname}.py
    vital/tests
    ${modname})
endforeach()
