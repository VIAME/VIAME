/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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
 * \brief Register VXL algorithms implementation
 */

#include <arrows/vxl/kwiver_algo_vxl_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/vxl/bundle_adjust.h>
#include <arrows/vxl/close_loops_homography_guided.h>
#include <arrows/vxl/estimate_canonical_transform.h>
#include <arrows/vxl/estimate_essential_matrix.h>
#include <arrows/vxl/estimate_fundamental_matrix.h>
#include <arrows/vxl/estimate_homography.h>
#include <arrows/vxl/estimate_similarity_transform.h>
#include <arrows/vxl/image_io.h>
#include <arrows/vxl/optimize_cameras.h>
#include <arrows/vxl/split_image.h>
#include <arrows/vxl/triangulate_landmarks.h>
#include <arrows/vxl/match_features_constrained.h>
#include <arrows/vxl/vidl_ffmpeg_video_input.h>


namespace kwiver {
namespace arrows {
namespace vxl {

extern "C"
KWIVER_ALGO_VXL_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.vxl" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory               implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::bundle_adjust );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (vpgl) to bundle adjust cameras and landmarks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl_homography_guided", kwiver::arrows::vxl::close_loops_homography_guided );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL to estimate a sequence of ground plane homographies to identify "
                       "frames to match for loop closure." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl_plane", kwiver::arrows::vxl::estimate_canonical_transform );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (rrel) to robustly estimate a ground plane for a canonical transform." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::estimate_essential_matrix );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (vpgl) to estimate an essential matrix." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::estimate_fundamental_matrix );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (vpgl) to estimate a fundamental matrix." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::estimate_homography );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (rrel) to robustly estimate a homography from matched features." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::estimate_similarity_transform );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (vpgl) to estimate a 3D similarity transformation between corresponding landmarks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::image_io );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (vil) to load and save image files." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::optimize_cameras );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (vpgl) to optimize camera parameters for fixed landmarks and tracks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::split_image );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Split a larger image into multiple smaller images" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl", kwiver::arrows::vxl::triangulate_landmarks );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (vpgl) to triangulate 3D landmarks from cameras and tracks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vxl_constrained", kwiver::arrows::vxl::match_features_constrained );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL to match descriptors under the constraints of similar geometry "
                       "(rotation, scale, position)." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "vidl_ffmpeg", kwiver::arrows::vxl::vidl_ffmpeg_video_input );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use VXL (vidl with FFMPEG) to read video files as a sequence of images." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  vpm.mark_module_as_loaded( module_name );
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
