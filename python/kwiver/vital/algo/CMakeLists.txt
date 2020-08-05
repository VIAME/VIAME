project(vital_python_algo)

include_directories(${pybind11_INCLUDE_DIR})

kwiver_add_python_library(algorithm
  vital/algo
  activity_detector.cxx
  activity_detector.h
  algorithm.cxx
  algorithm.h
  algorithm_module.cxx
  analyze_tracks.cxx
  analyze_tracks.h
  associate_detections_to_tracks.cxx
  associate_detections_to_tracks.h
  bundle_adjust.cxx
  bundle_adjust.h
  close_loops.cxx
  close_loops.h
  compute_association_matrix.cxx
  compute_association_matrix.h
  compute_depth.cxx
  compute_depth.h
  compute_ref_homography.cxx
  compute_ref_homography.h
  compute_stereo_depth_map.cxx
  compute_stereo_depth_map.h
  compute_track_descriptors.cxx
  compute_track_descriptors.h
  convert_image.cxx
  convert_image.h
  detect_features.cxx
  detect_features.h
  detect_motion.cxx
  detect_motion.h
  detected_object_filter.cxx
  detected_object_filter.h
  detected_object_set_input.cxx
  detected_object_set_input.h
  detected_object_set_output.cxx
  detected_object_set_output.h
  draw_detected_object_set.cxx
  draw_detected_object_set.h
  image_object_detector.cxx
  image_object_detector.h
  trampoline/activity_detector_trampoline.txx
  trampoline/algorithm_trampoline.txx
  trampoline/analyze_tracks_trampoline.txx
  trampoline/bundle_adjust_trampoline.txx
  trampoline/close_loops_trampoline.txx
  trampoline/compute_association_matrix_trampoline.txx
  trampoline/compute_depth_trampoline.txx
  trampoline/compute_ref_homography_trampoline.txx
  trampoline/compute_stereo_depth_map_trampoline.txx
  trampoline/compute_track_descriptors_trampoline.txx
  trampoline/convert_image_trampoline.txx
  trampoline/detect_features_trampoline.txx
  trampoline/detect_motion_trampoline.txx
  trampoline/detected_object_filter_trampoline.txx
  trampoline/detected_object_set_input_trampoline.txx
  trampoline/detected_object_set_output_trampoline.txx
  trampoline/draw_detected_object_set_trampoline.txx
  trampoline/image_object_detector_trampoline.txx
)


target_link_libraries(python-vital.algo-algorithm
  PUBLIC ${PYTHON_LIBRARIES}
          vital
          vital_config
          vital_algo
          vital_python_util
)

kwiver_add_python_library(algorithm_factory
  vital/algo
  algorithm_factory.cxx
)

target_link_libraries(python-vital.algo-algorithm_factory
  PUBLIC ${PYTHON_LIBRARIES}
          vital
          vital_algo
)

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
                                vital/algo
                                __init__)
