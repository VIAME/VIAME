# Build / Install plugin containing OpenCV algorithm implementations + support
# structures

set( ocv_headers_public
  analyze_tracks.h
  camera_intrinsics.h
  convert_color_space.h
  descriptor_set.h
  detect_features.h
  detect_features_AGAST.h
  detect_features_FAST.h
  detect_features_GFTT.h
  detect_features_MSD.h
  detect_features_MSER.h
  detect_features_simple_blob.h
  detect_features_STAR.h
  detect_motion_3frame_differencing.h
  detect_motion_mog2.h
  draw_detected_object_set.h
  draw_tracks.h
  estimate_fundamental_matrix.h
  estimate_homography.h
#  estimate_pnp.h
  extract_descriptors.h
  extract_descriptors_BRIEF.h
  extract_descriptors_DAISY.h
  extract_descriptors_FREAK.h
  extract_descriptors_LATCH.h
  extract_descriptors_LUCID.h
  feature_set.h
  feature_detect_extract_BRISK.h
  feature_detect_extract_ORB.h
  feature_detect_extract_SIFT.h
  feature_detect_extract_SURF.h
  detect_heat_map.h
  hough_circle_detector.h
  image_container.h
  image_io.h
  mat_image_memory.h
  match_features.h
  match_features_bruteforce.h
  match_features_flannbased.h
  match_set.h
  merge_images.h
  refine_detections_grabcut.h
  refine_detections_watershed.h
  refine_detections_write_to_disk.h
  split_image_channels.h
  split_image_horizontally.h
  track_features_klt.h
  windowed_detector.h
  windowed_detector_resize.h
  windowed_trainer.h
  )
kwiver_install_headers(
  SUBDIR     arrows/ocv
  ${ocv_headers_public}
  )
kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/kwiver_algo_ocv_export.h
  NOPATH   SUBDIR     arrows/ocv
  )

set( ocv_sources
  analyze_tracks.cxx
  camera_intrinsics.cxx
  convert_color_space.cxx
  descriptor_set.cxx
  detect_features.cxx
  detect_features_AGAST.cxx
  detect_features_FAST.cxx
  detect_features_GFTT.cxx
  detect_features_MSD.cxx
  detect_features_MSER.cxx
  detect_features_simple_blob.cxx
  detect_features_STAR.cxx
  detect_motion_3frame_differencing.cxx
  detect_motion_mog2.cxx
  draw_detected_object_set.cxx
  draw_tracks.cxx
  estimate_fundamental_matrix.cxx
  estimate_homography.cxx
#  estimate_pnp.cxx
  extract_descriptors.cxx
  extract_descriptors_BRIEF.cxx
  extract_descriptors_DAISY.cxx
  extract_descriptors_FREAK.cxx
  extract_descriptors_LATCH.cxx
  extract_descriptors_LUCID.cxx
  detect_heat_map.cxx
  hough_circle_detector.cxx
  feature_set.cxx
  feature_detect_extract_BRISK.cxx
  feature_detect_extract_ORB.cxx
  feature_detect_extract_SIFT.cxx
  feature_detect_extract_SURF.cxx
  image_container.cxx
  image_io.cxx
  mat_image_memory.cxx
  match_features.cxx
  match_features_bruteforce.cxx
  match_features_flannbased.cxx
  match_set.cxx
  merge_images.cxx
  refine_detections_grabcut.cxx
  refine_detections_util.cxx
  refine_detections_util.h
  refine_detections_watershed.cxx
  refine_detections_write_to_disk.cxx
  split_image_channels.cxx
  split_image_horizontally.cxx
  track_features_klt.cxx
  windowed_detector.cxx
  windowed_detector_resize.cxx
  windowed_trainer.cxx
  )

if( KWIVER_HAS_OPENCV_VER_3 )
  add_definitions(-DKWIVER_HAS_OPENCV_VER_3)
endif()

kwiver_add_library( kwiver_algo_ocv
  ${ocv_headers_public}
  ${ocv_sources}
  )
target_link_libraries( kwiver_algo_ocv
  PUBLIC               kwiver_algo_core
                       ${OpenCV_LIBS}
  PRIVATE              kwiversys
                       ${Boost_SYSTEM_LIBRARY}
                       ${Boost_FILESYSTEM_LIBRARY}
                       ${Boost_REGEX_LIBRARY}
  )

algorithms_create_plugin( kwiver_algo_ocv
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
