/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Algorithm and utility functions for adding keypoints to detections from masks
 */

#ifndef VIAME_OPENCV_ADD_KEYPOINTS_FROM_MASK_H
#define VIAME_OPENCV_ADD_KEYPOINTS_FROM_MASK_H

#include "viame_opencv_export.h"

#include <vital/algo/refine_detections.h>
#include <vital/types/detected_object.h>
#include <vital/types/bounding_box.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <opencv2/core/core.hpp>

#include <vector>
#include <utility>

namespace viame
{

// =============================================================================
// Utility functions for computing keypoints from detection masks
// =============================================================================

/// Extract mask points from a detection in image coordinates.
///
/// \param det The detection to extract mask points from
/// \return Vector of points in image coordinates, empty if no mask
VIAME_OPENCV_EXPORT
std::vector< cv::Point >
get_mask_points( kwiver::vital::detected_object_sptr det );

/// Compute oriented bounding box corner points from a detection's mask or bounding box.
///
/// If the detection has a mask, computes the minimum area rotated rectangle
/// from the mask's convex hull. Otherwise, returns the axis-aligned bounding
/// box corners.
///
/// \param det The detection to compute box points for
/// \return Vector of 4 corner points in image coordinates
VIAME_OPENCV_EXPORT
std::vector< cv::Point2d >
compute_box_points( kwiver::vital::detected_object_sptr det );

/// Compute head and tail keypoints from oriented bounding box points.
///
/// Given 4 corner points of an oriented bounding box, computes the midpoints
/// of the 4 edges and returns the two with the maximum and minimum x coordinates
/// as head and tail respectively.
///
/// \param box_points Vector of 4 corner points
/// \return Pair of (head, tail) points where head has max x and tail has min x
VIAME_OPENCV_EXPORT
std::pair< cv::Point2d, cv::Point2d >
center_keypoints( const std::vector< cv::Point2d >& box_points );

/// Add head and tail keypoints to a detection based on its mask or bounding box.
///
/// Convenience function that computes box points and center keypoints,
/// then adds them to the detection.
///
/// \param det The detection to add keypoints to (modified in place)
/// \return true if keypoints were added, false if detection has no valid geometry
VIAME_OPENCV_EXPORT
bool
add_keypoints_from_box( kwiver::vital::detected_object_sptr det );

/// Compute keypoints using oriented bounding box method.
///
/// Uses midpoints of the short edges of the minimum-area oriented bounding box.
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_OPENCV_EXPORT
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_oriented_bbox( kwiver::vital::detected_object_sptr det );

/// Compute keypoints using Principal Component Analysis.
///
/// Finds the major axis of the mask points using PCA, then returns the
/// extreme points along that axis.
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_OPENCV_EXPORT
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_pca( kwiver::vital::detected_object_sptr det );

/// Compute keypoints using farthest points method.
///
/// Finds the two points on the convex hull that are farthest apart
/// (polygon diameter).
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_OPENCV_EXPORT
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_farthest( kwiver::vital::detected_object_sptr det );

/// Compute keypoints using convex hull extremes method.
///
/// Computes the convex hull of the mask, finds its oriented bounding box,
/// and returns midpoints of the short edges.
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_OPENCV_EXPORT
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_hull_extremes( kwiver::vital::detected_object_sptr det );

/// Compute keypoints using skeleton/medial axis method.
///
/// Computes the medial axis/skeleton of the mask using morphological thinning,
/// then finds the endpoints. If multiple endpoints exist, selects the two
/// farthest apart.
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_OPENCV_EXPORT
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_skeleton( kwiver::vital::detected_object_sptr det );

/// Compute keypoints using specified method.
///
/// Convenience function that dispatches to the appropriate keypoint computation
/// method based on the method string.
///
/// \param det The detection to compute keypoints for
/// \param method Method name: "oriented_bbox", "pca", "farthest", "hull_extremes", or "skeleton"
/// \return Pair of (head, tail) points where head has max x
VIAME_OPENCV_EXPORT
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints( kwiver::vital::detected_object_sptr det, const std::string& method );

/// Check if a keypoint method string is valid.
///
/// \param method Method name to validate
/// \return true if method is valid, false otherwise
VIAME_OPENCV_EXPORT
bool
is_valid_keypoint_method( const std::string& method );

/// Get description string for keypoint method configuration.
///
/// \return Configuration description string listing all available methods
VIAME_OPENCV_EXPORT
std::string
keypoint_method_description();

// =============================================================================
// Algorithm class
// =============================================================================

/**
 * @brief Algorithm that adds head/tail keypoints to detections based on their
 *        mask or bounding box.
 *
 * This algorithm takes a detection set as input, computes keypoints using one
 * of several methods (oriented bounding box, PCA, farthest points, hull extremes,
 * or skeleton), and adds head/tail keypoints. The head keypoint is placed at
 * the end with the larger x coordinate.
 */
class VIAME_OPENCV_EXPORT add_keypoints_from_mask
  : public kwiver::vital::algo::refine_detections
{
public:
  PLUGGABLE_IMPL( add_keypoints_from_mask,
                  refine_detections,
                  "add_keypoints_from_mask",
                  "Adds head and tail keypoints to detections based on their "
                  "mask or bounding box using configurable methods.",
    PARAM_DEFAULT( method, std::string,
                   "Method for computing keypoints from polygon/mask. Options: "
                   "oriented_bbox (default), pca, farthest, hull_extremes, skeleton",
                   "oriented_bbox" )
  )

  virtual ~add_keypoints_from_mask() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

  virtual kwiver::vital::detected_object_set_sptr
  refine( kwiver::vital::image_container_sptr image_data,
          kwiver::vital::detected_object_set_sptr detections ) const override;

}; // end class add_keypoints_from_mask

} // end namespace viame

#endif // VIAME_OPENCV_ADD_KEYPOINTS_FROM_MASK_H
