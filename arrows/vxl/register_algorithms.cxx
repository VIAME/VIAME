// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Register VXL algorithms implementation
 */

#include <arrows/vxl/kwiver_algo_vxl_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/vxl/average_frames.h>
#include <arrows/vxl/bundle_adjust.h>
#include <arrows/vxl/close_loops_homography_guided.h>
#include <arrows/vxl/color_commonality_filter.h>
#include <arrows/vxl/convert_image.h>
#include <arrows/vxl/estimate_canonical_transform.h>
#include <arrows/vxl/estimate_essential_matrix.h>
#include <arrows/vxl/estimate_fundamental_matrix.h>
#include <arrows/vxl/estimate_homography.h>
#include <arrows/vxl/estimate_similarity_transform.h>
#include <arrows/vxl/high_pass_filter.h>
#include <arrows/vxl/image_io.h>
#include <arrows/vxl/optimize_cameras.h>
#include <arrows/vxl/pixel_feature_extractor.h>
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
    kwiver::vital::algorithm_registrar reg( vpm, "arrows.vxl" );

  if (reg.is_module_loaded())
  {
    return;
  }

  using namespace kwiver::arrows::vxl;

  reg.register_algorithm< average_frames >();
  reg.register_algorithm< bundle_adjust >();
  reg.register_algorithm< close_loops_homography_guided >();
  reg.register_algorithm< color_commonality_filter >();
  reg.register_algorithm< convert_image >();
  reg.register_algorithm< estimate_canonical_transform >();
  reg.register_algorithm< estimate_essential_matrix >();
  reg.register_algorithm< estimate_fundamental_matrix >();
  reg.register_algorithm< estimate_homography >();
  reg.register_algorithm< estimate_similarity_transform >();
  reg.register_algorithm< high_pass_filter >();
  reg.register_algorithm< image_io >();
  reg.register_algorithm< optimize_cameras >();
  reg.register_algorithm< pixel_feature_extractor >();
  reg.register_algorithm< split_image >();
  reg.register_algorithm< triangulate_landmarks >();
  reg.register_algorithm< match_features_constrained >();
  reg.register_algorithm< vidl_ffmpeg_video_input >();

  reg.mark_module_as_loaded();
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
