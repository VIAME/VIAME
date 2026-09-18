/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Algorithm and utility functions for adding keypoints to detections from masks
 *
 * OpenCV until P7-T04b; `image_kernels` since. The utilities are still here
 * rather than in python because `measure_objects_process` calls
 * `compute_keypoints` and `is_valid_keypoint_method` directly, and one
 * implementation reached from both is better than two that have to agree.
 */

#ifndef VIAME_SEGMENTATION_ADD_KEYPOINTS_FROM_MASK_H
#define VIAME_SEGMENTATION_ADD_KEYPOINTS_FROM_MASK_H

#include "viame_segmentation_export.h"

#include <viame/algorithm_framework/algo/refine_detections.h>
#include <viame/core_types/detected_object.h>
#include <viame/core_types/bounding_box.h>
#include <viame/algorithm_framework/plugin/pluggable_macro_magic.h>

#include <viame/core_types/vector.h>

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
VIAME_SEGMENTATION_EXPORT
std::vector< viame::vector_2d >
get_mask_points( viame::detected_object_sptr det );

/// Compute oriented bounding box corner points from a detection's mask or bounding box.
///
/// If the detection has a mask, computes the minimum area rotated rectangle
/// from the mask's convex hull. Otherwise, returns the axis-aligned bounding
/// box corners.
///
/// \param det The detection to compute box points for
/// \return Vector of 4 corner points in image coordinates
VIAME_SEGMENTATION_EXPORT
std::vector< viame::vector_2d >
compute_box_points( viame::detected_object_sptr det );

/// Compute head and tail keypoints from oriented bounding box points.
///
/// Given 4 corner points of an oriented bounding box, computes the midpoints
/// of the 4 edges and returns the two with the maximum and minimum x coordinates
/// as head and tail respectively.
///
/// \param box_points Vector of 4 corner points
/// \return Pair of (head, tail) points where head has max x and tail has min x
VIAME_SEGMENTATION_EXPORT
std::pair< viame::vector_2d, viame::vector_2d >
center_keypoints( const std::vector< viame::vector_2d >& box_points );

/// Add head and tail keypoints to a detection based on its mask or bounding box.
///
/// Convenience function that computes box points and center keypoints,
/// then adds them to the detection.
///
/// \param det The detection to add keypoints to (modified in place)
/// \return true if keypoints were added, false if detection has no valid geometry
VIAME_SEGMENTATION_EXPORT
bool
add_keypoints_from_box( viame::detected_object_sptr det );

/// Compute keypoints using oriented bounding box method.
///
/// Uses midpoints of the short edges of the minimum-area oriented bounding box.
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_SEGMENTATION_EXPORT
std::pair< viame::vector_2d, viame::vector_2d >
compute_keypoints_oriented_bbox( viame::detected_object_sptr det );

/// Compute keypoints using Principal Component Analysis.
///
/// Finds the major axis of the mask points using PCA, then returns the
/// extreme points along that axis.
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_SEGMENTATION_EXPORT
std::pair< viame::vector_2d, viame::vector_2d >
compute_keypoints_pca( viame::detected_object_sptr det );

/// Compute keypoints using farthest points method.
///
/// Finds the two points on the convex hull that are farthest apart
/// (polygon diameter).
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_SEGMENTATION_EXPORT
std::pair< viame::vector_2d, viame::vector_2d >
compute_keypoints_farthest( viame::detected_object_sptr det );

/// Compute keypoints using convex hull extremes method.
///
/// Computes the convex hull of the mask, finds its oriented bounding box,
/// and returns midpoints of the short edges.
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_SEGMENTATION_EXPORT
std::pair< viame::vector_2d, viame::vector_2d >
compute_keypoints_hull_extremes( viame::detected_object_sptr det );

/// Compute keypoints using skeleton/medial axis method.
///
/// Computes the medial axis/skeleton of the mask using morphological thinning,
/// then finds the endpoints. If multiple endpoints exist, selects the two
/// farthest apart.
///
/// \param det The detection to compute keypoints for
/// \return Pair of (head, tail) points where head has max x
VIAME_SEGMENTATION_EXPORT
std::pair< viame::vector_2d, viame::vector_2d >
compute_keypoints_skeleton( viame::detected_object_sptr det );

/// Compute keypoints using specified method.
///
/// Convenience function that dispatches to the appropriate keypoint computation
/// method based on the method string.
///
/// \param det The detection to compute keypoints for
/// \param method Method name: "oriented_bbox", "pca", "farthest", "hull_extremes", or "skeleton"
/// \return Pair of (head, tail) points where head has max x
VIAME_SEGMENTATION_EXPORT
std::pair< viame::vector_2d, viame::vector_2d >
compute_keypoints( viame::detected_object_sptr det, const std::string& method );

/// Clip a point to the nearest point on a detection's mask boundary.
///
/// Given a target point and a detection with a mask, finds the nearest
/// point on the mask contour (polygon boundary). The search considers all
/// points along contour edges, not just contour vertices.
///
/// If the detection has no mask or the mask has no valid contour, the
/// original point is returned unchanged.
///
/// \param target The point to clip (in image coordinates)
/// \param det The detection containing the mask
/// \return The nearest point on the mask boundary (in image coordinates)
VIAME_SEGMENTATION_EXPORT
viame::vector_2d
clip_point_to_mask_boundary( const viame::vector_2d& target,
                             viame::detected_object_sptr det );

/// Check if a keypoint method string is valid.
///
/// \param method Method name to validate
/// \return true if method is valid, false otherwise
VIAME_SEGMENTATION_EXPORT
bool
is_valid_keypoint_method( const std::string& method );

/// Get description string for keypoint method configuration.
///
/// \return Configuration description string listing all available methods
VIAME_SEGMENTATION_EXPORT
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
class VIAME_SEGMENTATION_EXPORT add_keypoints_from_mask
  : public viame::algo::refine_detections
{
public:
  PLUGGABLE_IMPL( add_keypoints_from_mask,
                  "Adds head and tail keypoints to detections based on their "
                  "mask or bounding box using configurable methods.",
    PARAM_DEFAULT( method, std::string,
                   "Method for computing keypoints from polygon/mask. Options: "
                   "oriented_bbox (default), pca, farthest, hull_extremes, skeleton",
                   "oriented_bbox" ),
    PARAM_DEFAULT( clip_to_mask, bool,
                   "If true, after computing keypoints, snap each keypoint to "
                   "the nearest point on the mask/polygon boundary contour.",
                   false )
  )

  virtual ~add_keypoints_from_mask() = default;

  virtual bool check_configuration( viame::config_block_sptr config ) const override;

  virtual viame::detected_object_set_sptr
  refine( viame::image_container_sptr image_data,
          viame::detected_object_set_sptr detections ) const override;

}; // end class add_keypoints_from_mask

} // end namespace viame

#endif // VIAME_SEGMENTATION_ADD_KEYPOINTS_FROM_MASK_H
