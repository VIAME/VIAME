/*ckwg +29
 * Copyright 2014-2016 by Kitware, Inc.
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

#include <arrows/core/kwiver_algo_core_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/core/close_loops_bad_frames_only.h>
#include <arrows/core/close_loops_exhaustive.h>
#include <arrows/core/close_loops_keyframe.h>
#include <arrows/core/close_loops_multi_method.h>
#include <arrows/core/compute_ref_homography_core.h>
#include <arrows/core/convert_image_bypass.h>
#include <arrows/core/estimate_canonical_transform.h>
#include <arrows/core/filter_features_magnitude.h>
#include <arrows/core/hierarchical_bundle_adjust.h>
#include <arrows/core/initialize_cameras_landmarks.h>
#include <arrows/core/match_features_fundamental_matrix.h>
#include <arrows/core/match_features_homography.h>
#include <arrows/core/track_features_core.h>
#include <arrows/core/triangulate_landmarks.h>
#include <arrows/core/detected_object_set_input_kw18.h>
#include <arrows/core/detected_object_set_output_kw18.h>
#include <arrows/core/detected_object_set_input_csv.h>
#include <arrows/core/detected_object_set_output_csv.h>

namespace kwiver {
namespace arrows {
namespace core {

extern "C"
KWIVER_ALGO_CORE_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "algorithms.core" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory                  implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "bad_frames_only", kwiver::arrows::core::close_loops_bad_frames_only );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Attempts short-term loop closure based on percentage of feature "
                    "points tracked.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "exhaustive", kwiver::arrows::core::close_loops_exhaustive );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Exhaustive matching of all frame pairs, "
                    "or all frames within a moving window" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "keyframe", kwiver::arrows::core::close_loops_keyframe );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Establishes keyframes matches to all keyframes")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "multi_method", kwiver::arrows::core::close_loops_multi_method );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Iteratively run multiple loop closure algorithms" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::compute_ref_homography_core );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Default online sequential-frame reference homography estimator" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "bypass", kwiver::arrows::core::convert_image_bypass );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Performs no conversion and returns the given image container." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core_pca", kwiver::arrows::core::estimate_canonical_transform );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "magnitude", kwiver::arrows::core::filter_features_magnitude );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "hierarchical", kwiver::arrows::core::hierarchical_bundle_adjust );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::initialize_cameras_landmarks );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "fundamental_matrix_guided", kwiver::arrows::core::match_features_fundamental_matrix );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "homography_guided", kwiver::arrows::core::match_features_homography );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::track_features_core );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::triangulate_landmarks );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "kw18", kwiver::arrows::core::detected_object_set_input_kw18 );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "kw18", kwiver::arrows::core::detected_object_set_output_kw18 );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "csv", kwiver::arrows::core::detected_object_set_input_csv );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "csv", kwiver::arrows::core::detected_object_set_output_csv );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
