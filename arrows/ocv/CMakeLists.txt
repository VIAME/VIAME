# Build / Install plugin containing OpenCV algorithm implementations + support
# structures

set( ocv_headers_public
  analyze_tracks.h
  descriptor_set.h
  detect_features.h
  detect_features_AGAST.h
  detect_features_FAST.h
  detect_features_GFTT.h
  detect_features_MSD.h
  detect_features_MSER.h
  detect_features_simple_blob.h
  detect_features_STAR.h
  draw_detected_object_set.h
  draw_tracks.h
  estimate_fundamental_matrix.h
  estimate_homography.h
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
  hough_circle_detector.h
  image_container.h
  image_io.h
  mat_image_memory.h
  match_features.h
  match_features_bruteforce.h
  match_features_flannbased.h
  match_set.h
  refine_detections_draw.h
  split_image.h
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
  descriptor_set.cxx
  detect_features.cxx
  detect_features_AGAST.cxx
  detect_features_FAST.cxx
  detect_features_GFTT.cxx
  detect_features_MSD.cxx
  detect_features_MSER.cxx
  detect_features_simple_blob.cxx
  detect_features_STAR.cxx
  draw_detected_object_set.cxx
  draw_tracks.cxx
  estimate_fundamental_matrix.cxx
  estimate_homography.cxx
  extract_descriptors.cxx
  extract_descriptors_BRIEF.cxx
  extract_descriptors_DAISY.cxx
  extract_descriptors_FREAK.cxx
  extract_descriptors_LATCH.cxx
  extract_descriptors_LUCID.cxx
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
  refine_detections_draw.cxx
  split_image.cxx
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
  )

algorithms_create_plugin( kwiver_algo_ocv
  register_algorithms.cxx
  )

if (KWIVER_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
