/*ckwg +29
 * Copyright 2014-2018 by Kitware, Inc.
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
 * \file
 * \brief Defaults plugin algorithm registration interface impl
 */

#include <arrows/core/kwiver_algo_core_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/core/associate_detections_to_tracks_threshold.h>
#include <arrows/core/class_probablity_filter.h>
#include <arrows/core/close_loops_bad_frames_only.h>
#include <arrows/core/close_loops_exhaustive.h>
#include <arrows/core/close_loops_keyframe.h>
#include <arrows/core/close_loops_multi_method.h>
#include <arrows/core/compute_association_matrix_from_features.h>
#include <arrows/core/compute_ref_homography_core.h>
#include <arrows/core/convert_image_bypass.h>
#include <arrows/core/detected_object_set_input_csv.h>
#include <arrows/core/detected_object_set_input_kw18.h>
#include <arrows/core/detected_object_set_output_csv.h>
#include <arrows/core/detected_object_set_output_kw18.h>
#include <arrows/core/dynamic_config_none.h>
#include <arrows/core/estimate_canonical_transform.h>
#include <arrows/core/example_detector.h>
#include <arrows/core/feature_descriptor_io.h>
#include <arrows/core/filter_features_magnitude.h>
#include <arrows/core/filter_features_scale.h>
#include <arrows/core/filter_tracks.h>
#include <arrows/core/handle_descriptor_request_core.h>
#include <arrows/core/hierarchical_bundle_adjust.h>
#include <arrows/core/initialize_cameras_landmarks.h>
#include <arrows/core/initialize_object_tracks_threshold.h>
#include <arrows/core/match_features_fundamental_matrix.h>
#include <arrows/core/match_features_homography.h>
#include <arrows/core/read_object_track_set_kw18.h>
#include <arrows/core/read_track_descriptor_set_csv.h>
#include <arrows/core/track_features_core.h>
#include <arrows/core/triangulate_landmarks.h>
#include <arrows/core/video_input_filter.h>
#include <arrows/core/video_input_image_list.h>
#include <arrows/core/video_input_pos.h>
#include <arrows/core/video_input_split.h>
#include <arrows/core/write_object_track_set_kw18.h>
#include <arrows/core/write_track_descriptor_set_csv.h>


namespace kwiver {
namespace arrows {
namespace core {

namespace {

static auto const module_name         = std::string{ "arrows.core" };
static auto const module_version      = std::string{ "1.0" };
static auto const module_organization = std::string{ "Kitware Inc." };

// ----------------------------------------------------------------------------
template <typename algorithm_t>
void register_algorithm( kwiver::vital::plugin_loader& vpm )
{
  using kvpf = kwiver::vital::plugin_factory;

  auto fact = vpm.ADD_ALGORITHM( algorithm_t::name, algorithm_t );
  fact->add_attribute( kvpf::PLUGIN_DESCRIPTION,  algorithm_t::description )
       .add_attribute( kvpf::PLUGIN_MODULE_NAME,  module_name )
       .add_attribute( kvpf::PLUGIN_VERSION,      module_version )
       .add_attribute( kvpf::PLUGIN_ORGANIZATION, module_organization )
       ;
}

}

// ----------------------------------------------------------------------------
extern "C"
KWIVER_ALGO_CORE_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  register_algorithm< associate_detections_to_tracks_threshold >( vpm );
  register_algorithm< class_probablity_filter >( vpm );
  register_algorithm< close_loops_bad_frames_only >( vpm );
  register_algorithm< close_loops_exhaustive >( vpm );
  register_algorithm< close_loops_keyframe >( vpm );
  register_algorithm< close_loops_multi_method >( vpm );
  register_algorithm< compute_association_matrix_from_features >( vpm );
  register_algorithm< compute_ref_homography_core >( vpm );
  register_algorithm< convert_image_bypass >( vpm );
  register_algorithm< detected_object_set_input_csv >( vpm );
  register_algorithm< detected_object_set_input_kw18 >( vpm );
  register_algorithm< detected_object_set_output_csv >( vpm );
  register_algorithm< detected_object_set_output_kw18 >( vpm );
  register_algorithm< dynamic_config_none >( vpm );
  register_algorithm< estimate_canonical_transform >( vpm );
  register_algorithm< example_detector >( vpm );
  register_algorithm< feature_descriptor_io >( vpm );
  register_algorithm< filter_features_magnitude >( vpm );
  register_algorithm< filter_features_scale >( vpm );
  register_algorithm< filter_tracks >( vpm );
  register_algorithm< handle_descriptor_request_core >( vpm );
  register_algorithm< hierarchical_bundle_adjust >( vpm );
  register_algorithm< initialize_cameras_landmarks >( vpm );
  register_algorithm< initialize_object_tracks_threshold >( vpm );
  register_algorithm< match_features_fundamental_matrix >( vpm );
  register_algorithm< match_features_homography >( vpm );
  register_algorithm< read_object_track_set_kw18 >( vpm );
  register_algorithm< read_track_descriptor_set_csv >( vpm );
  register_algorithm< track_features_core >( vpm );
  register_algorithm< triangulate_landmarks >( vpm );
  register_algorithm< video_input_filter >( vpm );
  register_algorithm< video_input_image_list >( vpm );
  register_algorithm< video_input_pos >( vpm );
  register_algorithm< video_input_split >( vpm );
  register_algorithm< write_object_track_set_kw18 >( vpm );
  register_algorithm< write_track_descriptor_set_csv >( vpm );

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
