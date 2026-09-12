// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

// The `kwiver.vital.algo` module: one loader call per bound interface.
//
// CMake wrote this with `file( GENERATE )` from a list of class names until
// P8-T02 deleted the castxml generator that produced the bindings. It is
// source now, and adding an interface means adding two lines here.

#include <pybind11/pybind11.h>
#include "algorithm_python.h"
#include "algorithm_capabilities_python.h"
#include "associate_detections_to_tracks_python.h"
#include "close_loops_python.h"
#include "compute_ref_homography_python.h"
#include "compute_stereo_depth_map_python.h"
#include "compute_track_descriptors_python.h"
#include "detect_features_python.h"
#include "detect_motion_python.h"
#include "detected_object_filter_python.h"
#include "detected_object_set_input_python.h"
#include "detected_object_set_output_python.h"
#include "draw_detected_object_set_python.h"
#include "estimate_fundamental_matrix_python.h"
#include "estimate_homography_python.h"
#include "extract_descriptors_python.h"
#include "feature_descriptor_io_python.h"
#include "filter_features_python.h"
#include "image_filter_python.h"
#include "image_io_python.h"
#include "image_object_detector_python.h"
#include "initialize_object_tracks_python.h"
#include "match_descriptor_sets_python.h"
#include "match_features_python.h"
#include "merge_detections_python.h"
#include "merge_images_python.h"
#include "optimize_cameras_python.h"
#include "perform_text_query_python.h"
#include "read_object_track_set_python.h"
#include "read_track_descriptor_set_python.h"
#include "refine_detections_python.h"
#include "refine_tracks_python.h"
#include "resection_camera_python.h"
#include "segment_via_points_python.h"
#include "split_image_python.h"
#include "track_features_python.h"
#include "track_objects_python.h"
#include "train_detector_python.h"
#include "train_tracker_python.h"
#include "transform_2d_io_python.h"
#include "video_input_python.h"
#include "video_output_python.h"
#include "warp_image_python.h"
#include "write_object_track_set_python.h"
#include "write_track_descriptor_set_python.h"
#include "extract_descriptors_extras_python.h"
#include "estimator_extras_python.h"
namespace kwiver::vital::python {
PYBIND11_MODULE(algos,m)
{
   algorithm(m);
   algorithm_capabilities(m);
   associate_detections_to_tracks(m);
   close_loops(m);
   compute_ref_homography(m);
   compute_stereo_depth_map(m);
   compute_track_descriptors(m);
   detect_features(m);
   detect_motion(m);
   detected_object_filter(m);
   detected_object_set_input(m);
   detected_object_set_output(m);
   draw_detected_object_set(m);
   estimate_fundamental_matrix(m);
   estimate_homography(m);
   extract_descriptors(m);
   feature_descriptor_io(m);
   filter_features(m);
   image_filter(m);
   image_io(m);
   image_object_detector(m);
   initialize_object_tracks(m);
   match_descriptor_sets(m);
   match_features(m);
   merge_detections(m);
   merge_images(m);
   optimize_cameras(m);
   perform_text_query(m);
   read_object_track_set(m);
   read_track_descriptor_set(m);
   refine_detections(m);
   refine_tracks(m);
   resection_camera(m);
   segment_via_points(m);
   split_image(m);
   track_features(m);
   track_objects(m);
   train_detector(m);
   train_tracker(m);
   transform_2d_io(m);
   video_input(m);
   video_output(m);
   warp_image(m);
   write_object_track_set(m);
   write_track_descriptor_set(m);
   extract_descriptors_extras(m);
   estimator_extras(m);
   optimize_cameras_extras(m);
}
} // namespace 
