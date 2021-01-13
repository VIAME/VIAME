// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file algorithm_implementation.cxx
 *
 * \brief python bindings for algorithm
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <python/kwiver/vital/algo/trampoline/activity_detector_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/analyze_tracks_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/associate_detections_to_tracks_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/bundle_adjust_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/close_loops_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/compute_association_matrix_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/compute_depth_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/compute_ref_homography_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/compute_stereo_depth_map_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/convert_image_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/detect_features_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/detect_motion_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/detected_object_filter_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/detected_object_set_input_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/detected_object_set_output_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/draw_detected_object_set_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/draw_tracks_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/estimate_canonical_transform_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/estimate_essential_matrix_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/estimate_fundamental_matrix_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/estimate_homography_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/estimate_pnp_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/estimate_similarity_transform_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/extract_descriptors_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/feature_descriptor_io_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/filter_features_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/filter_tracks_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/image_filter_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/image_io_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/image_object_detector_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/initialize_cameras_landmarks_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/initialize_object_tracks_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/integrate_depth_maps_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/interpolate_track_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/keyframe_selection_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/match_descriptor_sets_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/match_features_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/merge_images_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/optimize_cameras_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/read_object_track_set_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/read_track_descriptor_set_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/refine_detections_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/split_image_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/track_features_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/train_detector_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/transform_2d_io_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/triangulate_landmarks_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/uuid_factory_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/uv_unwrap_mesh_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/video_input_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/write_object_track_set_trampoline.txx>
#include <python/kwiver/vital/algo/trampoline/write_track_descriptor_set_trampoline.txx>
#include <python/kwiver/vital/algo/algorithm.h>
#include <python/kwiver/vital/algo/activity_detector.h>
#include <python/kwiver/vital/algo/analyze_tracks.h>
#include <python/kwiver/vital/algo/associate_detections_to_tracks.h>
#include <python/kwiver/vital/algo/bundle_adjust.h>
#include <python/kwiver/vital/algo/close_loops.h>
#include <python/kwiver/vital/algo/compute_association_matrix.h>
#include <python/kwiver/vital/algo/compute_depth.h>
#include <python/kwiver/vital/algo/compute_ref_homography.h>
#include <python/kwiver/vital/algo/compute_stereo_depth_map.h>
#include <python/kwiver/vital/algo/convert_image.h>
#include <python/kwiver/vital/algo/detect_features.h>
#include <python/kwiver/vital/algo/detect_motion.h>
#include <python/kwiver/vital/algo/detected_object_filter.h>
#include <python/kwiver/vital/algo/detected_object_set_input.h>
#include <python/kwiver/vital/algo/detected_object_set_output.h>
#include <python/kwiver/vital/algo/draw_detected_object_set.h>
#include <python/kwiver/vital/algo/draw_tracks.h>
#include <python/kwiver/vital/algo/estimate_canonical_transform.h>
#include <python/kwiver/vital/algo/estimate_essential_matrix.h>
#include <python/kwiver/vital/algo/estimate_fundamental_matrix.h>
#include <python/kwiver/vital/algo/estimate_homography.h>
#include <python/kwiver/vital/algo/estimate_pnp.h>
#include <python/kwiver/vital/algo/estimate_similarity_transform.h>
#include <python/kwiver/vital/algo/extract_descriptors.h>
#include <python/kwiver/vital/algo/feature_descriptor_io.h>
#include <python/kwiver/vital/algo/filter_features.h>
#include <python/kwiver/vital/algo/filter_tracks.h>
#include <python/kwiver/vital/algo/image_filter.h>
#include <python/kwiver/vital/algo/image_io.h>
#include <python/kwiver/vital/algo/image_object_detector.h>
#include <python/kwiver/vital/algo/initialize_cameras_landmarks.h>
#include <python/kwiver/vital/algo/initialize_object_tracks.h>
#include <python/kwiver/vital/algo/integrate_depth_maps.h>
#include <python/kwiver/vital/algo/interpolate_track.h>
#include <python/kwiver/vital/algo/keyframe_selection.h>
#include <python/kwiver/vital/algo/match_descriptor_sets.h>
#include <python/kwiver/vital/algo/match_features.h>
#include <python/kwiver/vital/algo/merge_images.h>
#include <python/kwiver/vital/algo/optimize_cameras.h>
#include <python/kwiver/vital/algo/read_object_track_set.h>
#include <python/kwiver/vital/algo/read_track_descriptor_set.h>
#include <python/kwiver/vital/algo/refine_detections.h>
#include <python/kwiver/vital/algo/split_image.h>
#include <python/kwiver/vital/algo/track_features.h>
#include <python/kwiver/vital/algo/train_detector.h>
#include <python/kwiver/vital/algo/transform_2d_io.h>
#include <python/kwiver/vital/algo/triangulate_landmarks.h>
#include <python/kwiver/vital/algo/uuid_factory.h>
#include <python/kwiver/vital/algo/uv_unwrap_mesh.h>
#include <python/kwiver/vital/algo/video_input.h>
#include <python/kwiver/vital/algo/write_object_track_set.h>
#include <python/kwiver/vital/algo/write_track_descriptor_set.h>
#include <sstream>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;
using namespace kwiver::vital::python;

PYBIND11_MODULE(algos, m)
{
  algorithm(m);
  register_algorithm<kwiver::vital::algo::activity_detector,
            algorithm_def_ad_trampoline<>>(m, "activity_detector");
  register_algorithm<kwiver::vital::algo::analyze_tracks,
            algorithm_def_at_trampoline<>>(m, "analyze_tracks");
  register_algorithm<kwiver::vital::algo::associate_detections_to_tracks,
            algorithm_def_adtt_trampoline<>>(m, "associate_detections_to_tracks");
  register_algorithm<kwiver::vital::algo::bundle_adjust,
            algorithm_def_ba_trampoline<>>(m, "bundle_adjust");
  register_algorithm<kwiver::vital::algo::close_loops,
            algorithm_def_cl_trampoline<>>(m, "close_loops");
  register_algorithm<kwiver::vital::algo::compute_association_matrix,
            algorithm_def_cam_trampoline<>>(m, "compute_association_matrix");
  register_algorithm<kwiver::vital::algo::compute_depth,
            algorithm_def_cd_trampoline<>>(m, "compute_depth");
  register_algorithm<kwiver::vital::algo::compute_ref_homography,
            algorithm_def_crh_trampoline<>>(m, "compute_ref_homography");
  register_algorithm<kwiver::vital::algo::compute_stereo_depth_map,
            algorithm_def_csdm_trampoline<>>(m, "compute_stereo_depth_map");
  register_algorithm<kwiver::vital::algo::convert_image,
            algorithm_def_ci_trampoline<>>(m, "convert_image");
  register_algorithm<kwiver::vital::algo::detected_object_filter,
            algorithm_def_dof_trampoline<>>(m, "detected_object_filter");
  register_algorithm<kwiver::vital::algo::detected_object_set_input,
            algorithm_def_dosi_trampoline<>>(m, "detected_object_set_input");
  register_algorithm<kwiver::vital::algo::detected_object_set_output,
            algorithm_def_doso_trampoline<>>(m, "detected_object_set_output");
  register_algorithm<kwiver::vital::algo::detect_features,
            algorithm_def_df_trampoline<>>(m, "detect_features");
  register_algorithm<kwiver::vital::algo::detect_motion,
            algorithm_def_dm_trampoline<>>(m, "detect_motion");
  register_algorithm<kwiver::vital::algo::draw_detected_object_set,
            algorithm_def_ddos_trampoline<>>(m, "draw_detected_object_set");
  register_algorithm<kwiver::vital::algo::draw_tracks,
            algorithm_def_dt_trampoline<>>(m, "draw_tracks");
  register_algorithm<kwiver::vital::algo::estimate_canonical_transform,
            algorithm_def_ect_trampoline<>>(m, "estimate_canonical_transform");
  register_algorithm<kwiver::vital::algo::estimate_essential_matrix,
            algorithm_def_eem_trampoline<>>(m, "estimate_essential_matrix");
  register_algorithm<kwiver::vital::algo::estimate_fundamental_matrix,
            algorithm_def_efm_trampoline<>>(m, "estimate_fundamental_matrix");
  register_algorithm<kwiver::vital::algo::estimate_homography,
            algorithm_def_eh_trampoline<>>(m, "estimate_homography");
  register_algorithm<kwiver::vital::algo::estimate_pnp,
            algorithm_def_epnp_trampoline<>>(m, "estimate_pnp");
  register_algorithm<kwiver::vital::algo::estimate_similarity_transform,
            algorithm_def_est_trampoline<>>(m, "estimate_similarity_transform");
  register_algorithm<kwiver::vital::algo::extract_descriptors,
            algorithm_def_ed_trampoline<>>(m, "extract_descriptors");
  register_algorithm<kwiver::vital::algo::feature_descriptor_io,
            algorithm_def_fdio_trampoline<>>(m, "feature_descriptor_io");
  register_algorithm<kwiver::vital::algo::filter_features,
            algorithm_def_ff_trampoline<>>(m, "filter_features");
  register_algorithm<kwiver::vital::algo::filter_tracks,
            algorithm_def_ft_trampoline<>>(m, "filter_tracks");
  register_algorithm<kwiver::vital::algo::image_filter,
            algorithm_def_if_trampoline<>>(m, "image_filter");
  register_algorithm<kwiver::vital::algo::image_io,
            algorithm_def_iio_trampoline<>>(m, "image_io");
  register_algorithm<kwiver::vital::algo::image_object_detector,
            algorithm_def_iod_trampoline<>>(m, "image_object_detector");
  register_algorithm<kwiver::vital::algo::initialize_cameras_landmarks,
            algorithm_def_icl_trampoline<>>(m, "initialize_cameras_landmarks");
  register_algorithm<kwiver::vital::algo::initialize_object_tracks,
            algorithm_def_iot_trampoline<>>(m, "initialize_object_tracks");
  register_algorithm<kwiver::vital::algo::integrate_depth_maps,
            algorithm_def_idm_trampoline<>>(m, "integrate_depth_maps");
  register_algorithm<kwiver::vital::algo::interpolate_track,
            algorithm_def_it_trampoline<>>(m, "interpolate_track");
  register_algorithm<kwiver::vital::algo::keyframe_selection,
            algorithm_def_kf_trampoline<>>(m, "keyframe_selection");
  register_algorithm<kwiver::vital::algo::match_descriptor_sets,
            algorithm_def_mds_trampoline<>>(m, "match_descriptor_sets");
  register_algorithm<kwiver::vital::algo::match_features,
            algorithm_def_mf_trampoline<>>(m, "match_features");
  register_algorithm<kwiver::vital::algo::merge_images,
            algorithm_def_mi_trampoline<>>(m, "merge_images");
  register_algorithm<kwiver::vital::algo::optimize_cameras,
            algorithm_def_oc_trampoline<>>(m, "optimize_cameras");
  register_algorithm<kwiver::vital::algo::read_object_track_set,
            algorithm_def_rots_trampoline<>>(m, "read_object_track_set");
  register_algorithm<kwiver::vital::algo::read_track_descriptor_set,
            algorithm_def_rtds_trampoline<>>(m, "read_track_descriptor_set");
  register_algorithm<kwiver::vital::algo::refine_detections,
            algorithm_def_rd_trampoline<>>(m, "refine_detections");
  register_algorithm<kwiver::vital::algo::split_image,
            algorithm_def_si_trampoline<>>(m, "split_image");
  register_algorithm<kwiver::vital::algo::track_features,
            algorithm_def_tf_trampoline<>>(m, "track_features");
  register_algorithm<kwiver::vital::algo::train_detector,
            algorithm_def_td_trampoline<>>(m, "train_detector");
  register_algorithm<kwiver::vital::algo::transform_2d_io,
            algorithm_def_t2dio_trampoline<>>(m, "transform_2d_io");
  register_algorithm<kwiver::vital::algo::triangulate_landmarks,
            algorithm_def_tl_trampoline<>>(m, "triangulate_landmarks");
  register_algorithm<kwiver::vital::algo::uuid_factory,
            algorithm_def_uf_trampoline<>>(m, "uuid_factory");
  register_algorithm<kwiver::vital::algo::uv_unwrap_mesh,
            algorithm_def_uvum_trampoline<>>(m, "uv_unwrap_mesh");
  register_algorithm<kwiver::vital::algo::video_input,
            algorithm_def_vi_trampoline<>>(m, "video_input");
  register_algorithm<kwiver::vital::algo::write_object_track_set,
            algorithm_def_wots_trampoline<>>(m, "write_object_track_set");
  register_algorithm<kwiver::vital::algo::write_track_descriptor_set,
            algorithm_def_wtds_trampoline<>>(m, "write_track_descriptor_set");


  activity_detector(m);
  analyze_tracks(m);
  associate_detections_to_tracks(m);
  bundle_adjust(m);
  close_loops(m);
  compute_association_matrix(m);
  compute_depth(m);
  compute_ref_homography(m);
  compute_stereo_depth_map(m);
  convert_image(m);
  detected_object_filter(m);
  detected_object_set_input(m);
  detected_object_set_output(m);
  detect_features(m);
  detect_motion(m);
  draw_detected_object_set(m);
  draw_tracks(m);
  estimate_canonical_transform(m);
  estimate_essential_matrix(m);
  estimate_fundamental_matrix(m);
  estimate_homography(m);
  estimate_pnp(m);
  estimate_similarity_transform(m);
  extract_descriptors(m);
  feature_descriptor_io(m);
  filter_features(m);
  filter_tracks(m);
  image_filter(m);
  image_io(m);
  image_object_detector(m);
  initialize_cameras_landmarks(m);
  initialize_object_tracks(m);
  integrate_depth_maps(m);
  interpolate_track(m);
  keyframe_selection(m);
  match_descriptor_sets(m);
  match_features(m);
  merge_images(m);
  optimize_cameras(m);
  read_object_track_set(m);
  read_track_descriptor_set(m);
  refine_detections(m);
  split_image(m);
  track_features(m);
  train_detector(m);
  transform_2d_io(m);
  triangulate_landmarks(m);
  uuid_factory(m);
  uv_unwrap_mesh(m);
  video_input(m);
  write_object_track_set(m);
  write_track_descriptor_set(m);
}
}
}
}
