# In-house image processing algorithms
#
# These replace the third-party implementations one at a time, keeping the
# registered names and config keys. Nothing here may include OpenCV or VXL.

set( plugin_headers
  average_frames.h
  close_loops_homography_guided.h
  color_commonality.h
  convert_image.h
  equalize_via_percentiles.h
  morphology.h
  threshold.h

  # Imported from arrows/ocv and arrows/core in P5-T04. Unchanged apart from
  # their include paths and their export macro, which is this library's now;
  # the namespace is still kwiver's, and phase 8 is where that changes.
  close_loops_appearance_indexed.h
  close_loops_bad_frames_only.h
  compute_ref_homography_core.h
  detect_features_filtered.h
  draw_detected_object_set.h
  filter_features_nonmax.h
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
  equalize_via_percentiles.cxx
  morphology.cxx
  threshold.cxx

  close_loops_appearance_indexed.cxx
  close_loops_bad_frames_only.cxx
  compute_ref_homography_core.cxx
  detect_features_filtered.cxx
  draw_detected_object_set.cxx
  filter_features_nonmax.cxx
  match_features_homography.cxx
  merge_images.cxx
  refine_detections_write_to_disk.cxx
  split_image.cxx
  split_image_channels.cxx
  track_features_core.cxx
  track_set_impl.cxx
  )

# `viame/image_processing`, not `viame`: the tree includes these as
# `<viame/image_processing/morphology.h>` through the `library-include`
# symlink, and an installed consumer has to find them at the same path.
viame_install_headers(
  SUBDIR     viame/image_processing
  ${plugin_headers}
  )

viame_install_headers(
  ${CMAKE_CURRENT_BINARY_DIR}/viame_image_processing_export.h
  NOPATH   SUBDIR     viame/image_processing
  )

viame_add_library( viame_image_processing
  ${plugin_headers}
  ${plugin_sources}
  )

target_include_directories( viame_image_processing PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

target_link_libraries( viame_image_processing
  PUBLIC               kwiver::vital
                       kwiver::vital_algo
                       kwiver::vital_config
                       kwiver::vital_exceptions
                       kwiver::vital_logger
                       kwiver::vital_util
                       viame_image_ops
                       # codecs/, for the detection chips ocv_write saves
                       viame_video_io
  )


set_target_properties( viame_image_processing PROPERTIES
  SOVERSION            ${VIAME_VERSION_MAJOR}
  )

viame_register_statically( viame_image_processing
  register_algorithms.cxx
  )



###
# Python implementations
##
if( VIAME_ENABLE_PYTHON )
  # Everything in the directory: the OpenCV replacements that came off
  # arrows/ocv in P7, and the multi-camera homography, alignment, optical
  # flow and percentile-normalisation modules that came from `plugins/core`
  # in P2-T05.
  viame_add_python_package( image_processing )
endif()

###
# Processes
#
# Imported from sprokit/processes/core in P5-T04.
##
set( process_headers
  accumulate_image_statistics_process.h
  align_multimodal_imagery_process.h
  draw_detected_object_set_process.h
  image_filter_process.h
  merge_images_process.h
  split_image_process.h
  stabilize_image_process.h
  stack_frames_process.h
  warp_detections_process.h
  warp_image_process.h
  )

set( process_sources
  register_processes.cxx
  accumulate_image_statistics_process.cxx
  align_multimodal_imagery_process.cxx
  draw_detected_object_set_process.cxx
  image_filter_process.cxx
  merge_images_process.cxx
  split_image_process.cxx
  stabilize_image_process.cxx
  stack_frames_process.cxx
  warp_detections_process.cxx
  warp_image_process.cxx
  )

viame_add_plugin( viame_processes_image_processing
  KIND        PROCESSES
  SOURCES          ${process_sources}
                   ${process_headers}
  PRIVATE          kwiver::sprokit_pipeline
                   kwiver::vital
                   kwiver::vital_vpm
                   kwiver::vital_logger
                   kwiver::vital_config
                   kwiver::vital_util
                   kwiver::vital_exceptions
                   kwiver::vital_algo
                   viame_image_processing
  )

target_include_directories( viame_processes_image_processing PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )
