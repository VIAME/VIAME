/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * \brief Plugin algorithm registration for MVG Arrow
 */

#include <arrows/mvg/kwiver_algo_mvg_plugin_export.h>

#include <vital/algo/algorithm_factory.h>

#include <arrows/mvg/algo/hierarchical_bundle_adjust.h>
#include <arrows/mvg/algo/initialize_cameras_landmarks.h>
#include <arrows/mvg/algo/initialize_cameras_landmarks_basic.h>
#include <arrows/mvg/algo/triangulate_landmarks.h>

// TODO: These are files that should move to MVG from Core
//#include <arrows/mvg/close_loops_bad_frames_only.h>
//#include <arrows/mvg/close_loops_appearance_indexed.h>
//#include <arrows/mvg/close_loops_exhaustive.h>
//#include <arrows/mvg/close_loops_keyframe.h>
//#include <arrows/mvg/close_loops_multi_method.h>
//#include <arrows/mvg/compute_ref_homography_core.h>
//#include <arrows/mvg/estimate_canonical_transform.h>
//#include <arrows/mvg/feature_descriptor_io.h>
//#include <arrows/mvg/filter_features_magnitude.h>
//#include <arrows/mvg/filter_features_scale.h>
//#include <arrows/mvg/filter_tracks.h>
//#include <arrows/mvg/keyframe_selector_basic.h>
//#include <arrows/mvg/match_features_fundamental_matrix.h>
//#include <arrows/mvg/match_features_homography.h>
//#include <arrows/mvg/track_features_augment_keyframes.h>
//#include <arrows/mvg/track_features_core.h>


namespace kwiver {
namespace arrows {
namespace mvg {


// ----------------------------------------------------------------------------
extern "C"
KWIVER_ALGO_MVG_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::vital::algorithm_registrar reg( vpm, "arrows.mvg" );

  if (reg.is_module_loaded())
  {
    return;
  }

  reg.register_algorithm< hierarchical_bundle_adjust >();
  reg.register_algorithm< initialize_cameras_landmarks >();
  reg.register_algorithm< initialize_cameras_landmarks_basic >();
  reg.register_algorithm< triangulate_landmarks >();

  reg.mark_module_as_loaded();
}

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver
