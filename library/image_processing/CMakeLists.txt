# In-house image processing algorithms
#
# These replace the third-party implementations one at a time, keeping the
# registered names and config keys. Nothing here may include OpenCV or VXL.

set( plugin_headers
  average_frames.h
  close_loops_homography_guided.h
  color_commonality.h
  convert_image.h
  morphology.h
  threshold.h

  # Imported from arrows/ocv and arrows/core in P5-T04. Unchanged apart from
  # their include paths and their export macro, which is this library's now;
  # the namespace is still kwiver's, and phase 8 is where that changes.
  close_loops_appearance_indexed.h
  close_loops_bad_frames_only.h
  compute_ref_homography_core.h
  detect_features.h
  detect_features_filtered.h
  draw_detected_object_set.h
  estimate_fundamental_matrix.h
  estimate_homography.h
  extract_descriptors.h
  feature_detect_extract_SIFT.h
  feature_detect_extract_SURF.h
  filter_features_nonmax.h
  match_features.h
  match_features_flannbased.h
  match_features_homography.h
  merge_images.h
  refine_detections_write_to_disk.h
  split_image.h
  split_image_channels.h
  track_features_core.h
  track_set_impl.h
  )

set( plugin_sources
  average_frames.cxx
  close_loops_homography_guided.cxx
  color_commonality.cxx
  convert_image.cxx
  morphology.cxx
  threshold.cxx

  close_loops_appearance_indexed.cxx
  close_loops_bad_frames_only.cxx
  compute_ref_homography_core.cxx
  detect_features.cxx
  detect_features_filtered.cxx
  draw_detected_object_set.cxx
  estimate_fundamental_matrix.cxx
  estimate_homography.cxx
  extract_descriptors.cxx
  feature_detect_extract_SIFT.cxx
  feature_detect_extract_SURF.cxx
  filter_features_nonmax.cxx
  match_features.cxx
  match_features_flannbased.cxx
  match_features_homography.cxx
  merge_images.cxx
  refine_detections_write_to_disk.cxx
  split_image.cxx
  split_image_channels.cxx
  track_features_core.cxx
  track_set_impl.cxx
  )

kwiver_install_headers(
  SUBDIR     viame
  ${plugin_headers}
  )

kwiver_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_image_processing_export.h
  NOPATH   SUBDIR     viame
  )

kwiver_add_library( viame_image_processing
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_image_processing PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  # OpenCV's headers, for the classes imported from arrows/ocv
  $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
  )

target_link_libraries( viame_image_processing
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       viame_image_ops
                       kwiver::kwiversys
  PRIVATE              ${OpenCV_LIBS}
                       # image_container and the rest of arrows/ocv's vital
                       # bridge; phase 7 deletes it
                       viame_opencv_bridge
  )

# The same guards kwiver set on its ocv arrow. The SIFT and SURF detectors
# and descriptors compile out entirely without them, taking ocv_SIFT and
# ocv_SURF with them; PUBLIC so the separately compiled plugin registration
# sees the same guard the headers do.
target_compile_definitions( viame_image_processing
  PUBLIC KWIVER_OPENCV_VERSION_MAJOR=${KWIVER_OPENCV_VERSION_MAJOR}
  PUBLIC HAVE_OPENCV_NONFREE
  PUBLIC HAVE_OPENCV_XFEATURES2D
  )

set_target_properties( viame_image_processing PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

algorithms_create_plugin( viame_image_processing
  register_algorithms.cxx
  )

target_link_libraries( viame_image_processing_plugin
  PUBLIC               kwiver::vital_vpm
  )


###
# Processes
#
# Imported from sprokit/processes/core in P5-T04.
##
set( process_headers
  draw_detected_object_set_process.h
  image_filter_process.h
  merge_images_process.h
  split_image_process.h
  stabilize_image_process.h
  )

set( process_sources
  register_processes.cxx
  draw_detected_object_set_process.cxx
  image_filter_process.cxx
  merge_images_process.cxx
  split_image_process.cxx
  stabilize_image_process.cxx
  )

kwiver_add_plugin( viame_processes_image_processing
  SUBDIR           ${viame_plugin_process_subdir}
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::kwiversys
                   kwiver::vital
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
                   viame_image_processing
  )

target_include_directories( viame_processes_image_processing PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  # OpenCV's headers, for the classes imported from arrows/ocv
  $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
  )
