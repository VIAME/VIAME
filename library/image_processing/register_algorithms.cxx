/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief In-house image processing algorithm registration
 */

#include "viame_image_processing_plugin_export.h"

#include <viame/algorithm_framework/algo/close_loops.h>
#include <viame/algorithm_framework/algo/compute_ref_homography.h>
#include <viame/algorithm_framework/algo/detect_features.h>
#include <viame/algorithm_framework/algo/draw_detected_object_set.h>
#include <viame/algorithm_framework/algo/estimate_fundamental_matrix.h>
#include <viame/algorithm_framework/algo/estimate_homography.h>
#include <viame/algorithm_framework/algo/extract_descriptors.h>
#include <viame/algorithm_framework/algo/filter_features.h>
#include <viame/algorithm_framework/algo/filter_tracks.h>
#include <viame/algorithm_framework/algo/image_filter.h>
#include <viame/algorithm_framework/algo/match_features.h>
#include <viame/algorithm_framework/algo/merge_images.h>
#include <viame/algorithm_framework/algo/refine_detections.h>
#include <viame/algorithm_framework/algo/split_image.h>
#include <viame/algorithm_framework/algo/track_features.h>
#include <viame/algorithm_framework/plugin/plugin_loader.h>

#include "average_frames.h"
#include "close_loops_homography_guided.h"
#include "color_commonality.h"
#include "convert_image.h"
#include "morphology.h"
#include "threshold.h"

// Imported from arrows/ocv and arrows/core in P5-T04
#include "close_loops_appearance_indexed.h"
#include "close_loops_bad_frames_only.h"
#include "compute_ref_homography_core.h"
#include "detect_features_filtered.h"
#include "filter_features_nonmax.h"
#include "filter_tracks.h"
#include "draw_detected_object_set.h"
#include "estimate_fundamental_matrix.h"
#include "estimate_homography.h"
#include "feature_detect_extract_SIFT.h"
#include "feature_detect_extract_SURF.h"
#include "match_features_flannbased.h"
#include "match_features_homography.h"
#include "merge_images.h"
#include "refine_detections_write_to_disk.h"
#include "split_image.h"
#include "split_image_channels.h"
#include "track_features_core.h"

namespace viame {

namespace kv = kwiver::vital;

extern "C"
VIAME_IMAGE_PROCESSING_PLUGIN_EXPORT
void
register_factories( kv::plugin_loader& vpm )
{
  using kvpf = kv::plugin_factory;
  const std::string module_name = "viame.image_processing";

  if( vpm.is_module_loaded( module_name ) )
  {
    return;
  }

#define VIAME_REGISTER_IMAGE_FILTER( impl )                              \
  {                                                                      \
    auto fact = vpm.add_factory< kv::algo::image_filter, impl >(          \
      impl::plugin_name() );                                             \
    fact->add_attribute( kvpf::PLUGIN_NAME, impl::plugin_name() )        \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,                          \
                      impl::plugin_description() );                      \
  }

  VIAME_REGISTER_IMAGE_FILTER( average_frames )
  VIAME_REGISTER_IMAGE_FILTER( color_commonality )
  VIAME_REGISTER_IMAGE_FILTER( convert_image )
  VIAME_REGISTER_IMAGE_FILTER( morphology )
  VIAME_REGISTER_IMAGE_FILTER( threshold )

  // The names arrows/vxl used to register, kept working now that it is gone.
  // Every one is checked against a recording of what the VXL implementation
  // produced; see tests/golden/vxl
#define VIAME_REGISTER_ALIAS( impl, alias )                              \
  {                                                                      \
    auto fact = vpm.add_factory< kv::algo::image_filter, impl >( alias ); \
    fact->add_attribute( kvpf::PLUGIN_NAME, alias )                      \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,                          \
                      impl::plugin_description() );                      \
  }

  VIAME_REGISTER_ALIAS( average_frames, "vxl_average" )
  VIAME_REGISTER_ALIAS( color_commonality, "vxl_color_commonality" )
  VIAME_REGISTER_ALIAS( convert_image, "vxl_convert_image" )
  VIAME_REGISTER_ALIAS( morphology, "vxl_morphology" )
  VIAME_REGISTER_ALIAS( threshold, "vxl_threshold" )

#undef VIAME_REGISTER_ALIAS
#undef VIAME_REGISTER_IMAGE_FILTER

  // Loop closure, which the image stabiliser selects. Registered here rather
  // than with the trackers because it works on frame to frame homographies
  {
    auto fact = vpm.add_factory< kv::algo::close_loops,
      close_loops_homography_guided >(
        close_loops_homography_guided::plugin_name() );
    fact->add_attribute( kvpf::PLUGIN_NAME,
                         close_loops_homography_guided::plugin_name() )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                      close_loops_homography_guided::plugin_description() );

    fact = vpm.add_factory< kv::algo::close_loops,
      close_loops_homography_guided >( "vxl_homography_guided" );
    fact->add_attribute( kvpf::PLUGIN_NAME, "vxl_homography_guided" )
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )
      .add_attribute( kvpf::PLUGIN_DESCRIPTION,
                      close_loops_homography_guided::plugin_description() );
  }


  // Imported from arrows/ocv and arrows/core in P5-T04, under the names
  // they registered under there.
#define VIAME_REGISTER_IMPORTED( interface, impl, plugin, blurb )        \
  {                                                                      \
    auto fact = vpm.add_factory< interface, impl >( plugin );            \
    fact->add_attribute( kvpf::PLUGIN_NAME, plugin )                     \
      .add_attribute( kvpf::PLUGIN_MODULE_NAME, module_name )            \
      .add_attribute( kvpf::PLUGIN_DESCRIPTION, blurb );                 \
  }

  VIAME_REGISTER_IMPORTED( kv::algo::estimate_fundamental_matrix,
                           kwiver::arrows::ocv::estimate_fundamental_matrix,
                           "ocv", "Estimate a fundamental matrix with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::estimate_homography,
                           kwiver::arrows::ocv::estimate_homography,
                           "ocv", "Estimate a homography with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::draw_detected_object_set,
                           kwiver::arrows::ocv::draw_detected_object_set,
                           "ocv", "Draw detected object sets on an image with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::match_features,
                           kwiver::arrows::ocv::match_features_flannbased,
                           "ocv_flann_based", "Match features with OpenCV's FLANN matcher" )

  VIAME_REGISTER_IMPORTED( kv::algo::detect_features,
                           kwiver::arrows::ocv::detect_features_SIFT,
                           "ocv_SIFT", "Detect SIFT features with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::extract_descriptors,
                           kwiver::arrows::ocv::extract_descriptors_SIFT,
                           "ocv_SIFT", "Extract SIFT descriptors with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::detect_features,
                           kwiver::arrows::ocv::detect_features_SURF,
                           "ocv_SURF", "Detect SURF features with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::extract_descriptors,
                           kwiver::arrows::ocv::extract_descriptors_SURF,
                           "ocv_SURF", "Extract SURF descriptors with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::refine_detections,
                           kwiver::arrows::ocv::refine_detections_write_to_disk,
                           "ocv_write", "Write detection chips to disk with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::split_image,
                           kwiver::arrows::ocv::split_image,
                           "ocv", "Split an image in half with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::split_image,
                           kwiver::arrows::ocv::split_image_channels,
                           "ocv_channels", "Split an image into its channels with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::merge_images,
                           kwiver::arrows::ocv::merge_images,
                           "ocv", "Merge two images with OpenCV" )

  VIAME_REGISTER_IMPORTED( kv::algo::compute_ref_homography,
                           kwiver::arrows::core::compute_ref_homography_core,
                           "core", "Compute a homography to a reference frame" )

  VIAME_REGISTER_IMPORTED( kv::algo::track_features,
                           kwiver::arrows::core::track_features_core,
                           "core", "Track features by detecting, describing and matching them" )

  VIAME_REGISTER_IMPORTED( kv::algo::match_features,
                           kwiver::arrows::core::match_features_homography,
                           "homography_guided",
                           "Match features and filter the matches by a homography" )

  VIAME_REGISTER_IMPORTED( kv::algo::detect_features,
                           kwiver::arrows::core::detect_features_filtered,
                           "filtered", "Detect features and filter them" )

  VIAME_REGISTER_IMPORTED( kv::algo::filter_features,
                           kwiver::arrows::core::filter_features_nonmax,
                           "nonmax", "Filter features by non-maximum suppression" )

  VIAME_REGISTER_IMPORTED( kv::algo::filter_tracks,
                           kwiver::arrows::core::filter_tracks,
                           "core", "Filter tracks by length and by match matrix importance" )

  VIAME_REGISTER_IMPORTED( kv::algo::close_loops,
                           kwiver::arrows::core::close_loops_bad_frames_only,
                           "bad_frames_only", "Close loops over runs of bad frames" )

  // Registered under `multi_method` with the appearance-indexed closer, as
  // kwiver registered it: see design/lite-findings.md 1.10. The class named
  // close_loops_multi_method was never registered by anything and did not
  // come across.
  VIAME_REGISTER_IMPORTED( kv::algo::close_loops,
                           kwiver::arrows::core::close_loops_appearance_indexed,
                           "multi_method", "Close loops by an appearance index" )

#undef VIAME_REGISTER_IMPORTED


  vpm.mark_module_as_loaded( module_name );
}

} // end namespace viame
