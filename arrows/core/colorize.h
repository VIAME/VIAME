// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for kwiver::arrows::core::colorize functions
 */

#ifndef KWIVER_ARROWS_COLORIZE_H_
#define KWIVER_ARROWS_COLORIZE_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/feature_set.h>
#include <vital/types/image_container.h>
#include <vital/types/landmark_map.h>
#include <vital/types/feature_track_set.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Extract feature colors from a frame image
/*
 * This function extracts the feature colors from a supplied frame image and
 * applies them to all features in a feature set by sampling the image at each
 * feature's location.
 *
 *  \param [in] features a set of features for which to assign colors
 *  \param [in] image the image from which to take colors
 *  \return a feature set with updated features
 */
KWIVER_ALGO_CORE_EXPORT
vital::feature_set_sptr extract_feature_colors(
  vital::feature_set const& features,
  vital::image_container const& image);

/// Extract feature colors from a frame image
/**
 * This function extracts the feature colors from a supplied frame image and
 * applies them to all features in the input track set with the same frame
 * number.
 *
 *  \param [in] tracks a set of feature tracks in which to colorize feature points
 *  \param [in] image the image from which to take colors
 *  \param [in] frame_id the frame number of the image
 *  \return a track set with updated features
 */
KWIVER_ALGO_CORE_EXPORT
vital::feature_track_set_sptr extract_feature_colors(
  vital::feature_track_set_sptr tracks,
  vital::image_container const& image,
  vital::frame_id_t frame_id);

/// Compute colors for landmarks
/**
 * This function computes landmark colors by taking the average color of all
 * associated feature points.
 *
 *  \param [in] landmarks a set of landmarks to be colored
 *  \param [in] tracks feature tracks to be used for computing landmark colors
 *  \return a set of colored landmarks
 */
KWIVER_ALGO_CORE_EXPORT
vital::landmark_map_sptr compute_landmark_colors(
  vital::landmark_map const& landmarks,
  vital::feature_track_set const& tracks);

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
