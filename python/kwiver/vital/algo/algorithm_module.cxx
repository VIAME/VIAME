/*ckwg +29
 * Copyright 2019-2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file algorithm_implementation.cxx
 *
 * \brief python bindings for algorithm
 */


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vital/algo/algorithm.h>
#include <vital/algo/analyze_tracks.h>
#include <vital/algo/image_object_detector.h>
#include <python/kwiver/vital/algo/activity_detector.h>
#include <python/kwiver/vital/algo/algorithm.h>
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
#include <python/kwiver/vital/algo/image_object_detector.h>
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
#include <python/kwiver/vital/algo/trampoline/image_object_detector_trampoline.txx>
#include <sstream>

namespace py = pybind11;

PYBIND11_MODULE(algorithm, m)
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
  register_algorithm<kwiver::vital::algo::image_object_detector,
            algorithm_def_iod_trampoline<>>(m, "image_object_detector");

  activity_detector(m);
  analyze_tracks(m);
  associate_detections_to_tracks(m);
  bundle_adjust(m);
  close_loops(m);
  compute_association_matrix(m);
  image_object_detector(m);
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
}
