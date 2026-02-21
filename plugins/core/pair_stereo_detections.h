/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo detection pairing utility functions
 */

#ifndef VIAME_CORE_PAIR_STEREO_DETECTIONS_H
#define VIAME_CORE_PAIR_STEREO_DETECTIONS_H

#include "viame_core_export.h"

#include <vital/types/vector.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image_container.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/detected_object.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/match_set.h>
#include <vital/logger/logger.h>

#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/match_features.h>
#include <vital/algo/estimate_homography.h>

#include <vector>
#include <utility>

namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

// Forward declaration
struct stereo_feature_correspondence;

// =============================================================================
// Configuration structures
// =============================================================================

/**
 * \brief Options for IOU-based stereo detection matching
 */
struct VIAME_CORE_EXPORT iou_matching_options
{
  double iou_threshold = 0.1;
  bool require_class_match = true;
  bool use_optimal_assignment = true;

  iou_matching_options() = default;
};

/**
 * \brief Options for calibration-based stereo detection matching
 */
struct VIAME_CORE_EXPORT calibration_matching_options
{
  double max_reprojection_error = 10.0;
  double default_depth = 5.0;
  bool require_class_match = true;
  bool use_optimal_assignment = true;

  calibration_matching_options() = default;
};

/**
 * \brief Options for feature-based stereo detection matching
 */
struct VIAME_CORE_EXPORT feature_matching_options
{
  int min_feature_match_count = 5;
  double min_feature_match_ratio = 0.1;
  bool use_homography_filtering = true;
  double homography_inlier_threshold = 5.0;
  double min_homography_inlier_ratio = 0.5;
  double box_expansion_factor = 1.1;
  bool require_class_match = true;
  bool use_optimal_assignment = true;

  feature_matching_options() = default;
};

/**
 * \brief Options for epipolar IOU-based stereo detection matching
 *
 * Projects left bounding box to right image using depth and camera geometry,
 * then matches based on IOU between projected box and right detection boxes.
 */
struct VIAME_CORE_EXPORT epipolar_iou_matching_options
{
  double iou_threshold = 0.1;
  double default_depth = 5.0;
  bool require_class_match = true;
  bool use_optimal_assignment = true;

  epipolar_iou_matching_options() = default;
};

/**
 * \brief Options for keypoint projection-based stereo detection matching
 *
 * When default_depth > 0: projects left head/tail keypoints to right image at
 * that depth, then matches based on pixel distance to right detection keypoints.
 * When default_depth <= 0: uses depth-independent epipolar line distance instead,
 * measuring how close right keypoints are to the epipolar lines of left keypoints.
 */
struct VIAME_CORE_EXPORT keypoint_projection_matching_options
{
  double max_keypoint_distance = 50.0;
  double default_depth = 0.0;
  bool require_class_match = true;
  bool use_optimal_assignment = true;

  keypoint_projection_matching_options() = default;
};

/**
 * \brief Algorithms required for feature-based matching
 */
struct VIAME_CORE_EXPORT feature_matching_algorithms
{
  kv::algo::detect_features_sptr feature_detector;
  kv::algo::extract_descriptors_sptr descriptor_extractor;
  kv::algo::match_features_sptr feature_matcher;
  kv::algo::estimate_homography_sptr homography_estimator;

  /// Check if all required algorithms are set
  bool is_valid() const
  {
    return feature_detector && descriptor_extractor && feature_matcher;
  }

  /// Check if homography filtering is available
  bool has_homography_estimator() const
  {
    return homography_estimator != nullptr;
  }
};

// =============================================================================
// Utility functions
// =============================================================================

/**
 * \brief Compute reprojection error for a pair of stereo correspondences
 *
 * Triangulates the 3D point from the stereo correspondences and computes
 * the RMS reprojection error back to both cameras.
 *
 * \param left_cam Left camera parameters
 * \param right_cam Right camera parameters
 * \param left_point 2D point in left image
 * \param right_point 2D point in right image
 * \return RMS reprojection error in pixels, or infinity if point is behind cameras
 */
VIAME_CORE_EXPORT double compute_stereo_reprojection_error(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  const kv::vector_2d& right_point );

/**
 * \brief Extract features within a detection's bounding box region
 *
 * Crops the image to the detection's bounding box (with optional expansion),
 * detects features, extracts descriptors, and transforms feature locations
 * back to full image coordinates.
 *
 * \param image Input image
 * \param bbox Bounding box region to extract features from
 * \param box_expansion_factor Factor to expand bounding box (1.0 = no expansion)
 * \param feature_detector Algorithm to detect features
 * \param descriptor_extractor Algorithm to extract descriptors
 * \param[out] features Detected features in full image coordinates
 * \param[out] descriptors Extracted descriptors
 */
VIAME_CORE_EXPORT void extract_detection_box_features(
  const kv::image_container_sptr& image,
  const kv::bounding_box_d& bbox,
  double box_expansion_factor,
  const kv::algo::detect_features_sptr& feature_detector,
  const kv::algo::extract_descriptors_sptr& descriptor_extractor,
  kv::feature_set_sptr& features,
  kv::descriptor_set_sptr& descriptors );

/**
 * \brief Filter feature matches using homography estimation
 *
 * Estimates a homography between matched feature points using RANSAC
 * and returns the inlier correspondences.
 *
 * \param features1 Features from first image
 * \param features2 Features from second image
 * \param matches Feature matches between the two images
 * \param homography_estimator Algorithm to estimate homography
 * \param inlier_threshold Maximum reprojection error for inliers
 * \param logger Optional logger for debug messages
 * \return Vector of inlier stereo correspondences
 */
VIAME_CORE_EXPORT std::vector< stereo_feature_correspondence > filter_matches_by_homography(
  const kv::feature_set_sptr& features1,
  const kv::feature_set_sptr& features2,
  const kv::match_set_sptr& matches,
  const kv::algo::estimate_homography_sptr& homography_estimator,
  double inlier_threshold,
  kv::logger_handle_t logger = nullptr );

/**
 * \brief Compute feature correspondences between two detection boxes
 *
 * Extracts features from both detection regions, matches them, and
 * optionally filters by homography consistency.
 *
 * \param det1 First detection
 * \param det2 Second detection
 * \param image1 Image containing first detection
 * \param image2 Image containing second detection
 * \param algorithms Feature matching algorithms
 * \param options Feature matching options
 * \param logger Optional logger for debug messages
 * \return Vector of stereo feature correspondences
 */
VIAME_CORE_EXPORT std::vector< stereo_feature_correspondence > compute_detection_feature_correspondences(
  const kv::detected_object_sptr& det1,
  const kv::detected_object_sptr& det2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  const feature_matching_algorithms& algorithms,
  const feature_matching_options& options,
  kv::logger_handle_t logger = nullptr );

/**
 * \brief Compute feature match score between two detections
 *
 * Returns a score where lower is better (like a cost), based on the
 * ratio of feature matches/inliers. Returns infinity if matching fails
 * or doesn't meet minimum thresholds.
 *
 * \param det1 First detection
 * \param det2 Second detection
 * \param image1 Image containing first detection
 * \param image2 Image containing second detection
 * \param algorithms Feature matching algorithms
 * \param options Feature matching options
 * \return Match score (lower is better), or infinity if no valid match
 */
VIAME_CORE_EXPORT double compute_detection_feature_match_score(
  const kv::detected_object_sptr& det1,
  const kv::detected_object_sptr& det2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  const feature_matching_algorithms& algorithms,
  const feature_matching_options& options );

/**
 * \brief Find stereo detection matches using IOU overlap
 *
 * Matches detections between left and right images based on bounding box
 * intersection-over-union.
 *
 * \param detections1 Detections from first (left) camera
 * \param detections2 Detections from second (right) camera
 * \param options Matching options
 * \return Vector of (index1, index2) pairs of matched detections
 */
VIAME_CORE_EXPORT std::vector< std::pair< int, int > > find_stereo_matches_iou(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const iou_matching_options& options );

/**
 * \brief Find stereo detection matches using camera calibration
 *
 * Matches detections by projecting left detection centers to right camera
 * using stereo geometry and comparing reprojection errors.
 *
 * \param detections1 Detections from first (left) camera
 * \param detections2 Detections from second (right) camera
 * \param left_cam Left camera parameters
 * \param right_cam Right camera parameters
 * \param options Matching options
 * \param logger Optional logger for error messages
 * \return Vector of (index1, index2) pairs of matched detections
 */
VIAME_CORE_EXPORT std::vector< std::pair< int, int > > find_stereo_matches_calibration(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const calibration_matching_options& options,
  kv::logger_handle_t logger = nullptr );

/**
 * \brief Find stereo detection matches using feature matching
 *
 * Matches detections by extracting and matching visual features within
 * detection bounding boxes.
 *
 * \param detections1 Detections from first (left) camera
 * \param detections2 Detections from second (right) camera
 * \param image1 Image from first (left) camera
 * \param image2 Image from second (right) camera
 * \param algorithms Feature matching algorithms
 * \param options Matching options
 * \param logger Optional logger for error messages
 * \return Vector of (index1, index2) pairs of matched detections
 */
VIAME_CORE_EXPORT std::vector< std::pair< int, int > > find_stereo_matches_feature(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  const feature_matching_algorithms& algorithms,
  const feature_matching_options& options,
  kv::logger_handle_t logger = nullptr );

/**
 * \brief Find stereo detection matches using epipolar IOU
 *
 * Projects left detection bounding boxes to right image using camera geometry
 * and a default depth, then matches based on IOU between projected boxes and
 * actual right detection bounding boxes.
 *
 * \param detections1 Detections from first (left) camera
 * \param detections2 Detections from second (right) camera
 * \param left_cam Left camera parameters
 * \param right_cam Right camera parameters
 * \param options Matching options
 * \param logger Optional logger for error messages
 * \return Vector of (index1, index2) pairs of matched detections
 */
VIAME_CORE_EXPORT std::vector< std::pair< int, int > > find_stereo_matches_epipolar_iou(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const epipolar_iou_matching_options& options,
  kv::logger_handle_t logger = nullptr );

/**
 * \brief Find stereo detection matches using keypoint projection
 *
 * Projects left head/tail keypoints to right image using camera geometry
 * and a default depth, then matches based on average pixel distance between
 * projected and actual right detection keypoints.
 *
 * \param detections1 Detections from first (left) camera
 * \param detections2 Detections from second (right) camera
 * \param left_cam Left camera parameters
 * \param right_cam Right camera parameters
 * \param options Matching options
 * \param logger Optional logger for error messages
 * \return Vector of (index1, index2) pairs of matched detections
 */
VIAME_CORE_EXPORT std::vector< std::pair< int, int > > find_stereo_matches_keypoint_projection(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const keypoint_projection_matching_options& options,
  kv::logger_handle_t logger = nullptr );

// =============================================================================
// Unified detection pairing dispatch
// =============================================================================

/**
 * \brief Parameters for the shared stereo detection pairing dispatch.
 *
 * This struct provides a single, method-agnostic interface for configuring
 * detection pairing.  Both pair_stereo_detections_process and
 * measure_objects_process use it so the method dispatch code lives in one place.
 */
struct VIAME_CORE_EXPORT detection_pairing_params
{
  /// Matching method: "iou", "calibration", "feature_matching",
  ///                  "epipolar_iou", or "keypoint_projection"
  std::string method = "iou";

  /// Generic threshold whose meaning depends on the method:
  ///   iou / epipolar_iou  → minimum IOU (default 0.1)
  ///   calibration         → max reprojection error in pixels (default 10.0)
  ///   keypoint_projection → max avg keypoint distance in pixels (default 50.0)
  ///   feature_matching    → (unused, score is internal)
  double threshold = 0.1;

  /// Default depth for projection-based methods (calibration, epipolar_iou,
  /// keypoint_projection).  0 = epipolar-line mode for keypoint_projection.
  double default_depth = 0.0;

  /// Only allow matches between detections with the same top class label
  bool require_class_match = true;

  /// Use greedy optimal assignment (true) vs simple sequential matching (false)
  bool use_optimal_assignment = true;

  detection_pairing_params() = default;
};

/**
 * \brief Dispatch stereo detection matching to the appropriate algorithm.
 *
 * \param params          Method name, threshold, and common options
 * \param detections1     Detections from left camera
 * \param detections2     Detections from right camera
 * \param left_cam        Left camera (may be nullptr for IOU / feature_matching)
 * \param right_cam       Right camera (may be nullptr for IOU / feature_matching)
 * \param image1          Left image  (required only for feature_matching)
 * \param image2          Right image (required only for feature_matching)
 * \param feature_algos   Feature algorithms (required only for feature_matching)
 * \param feature_opts    Feature options     (required only for feature_matching)
 * \param logger          Optional logger
 * \return Vector of (left_index, right_index) match pairs
 */
VIAME_CORE_EXPORT std::vector< std::pair< int, int > > find_stereo_detection_matches(
  const detection_pairing_params& params,
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::simple_camera_perspective* left_cam = nullptr,
  const kv::simple_camera_perspective* right_cam = nullptr,
  const kv::image_container_sptr& image1 = nullptr,
  const kv::image_container_sptr& image2 = nullptr,
  const feature_matching_algorithms* feature_algos = nullptr,
  const feature_matching_options* feature_opts = nullptr,
  kv::logger_handle_t logger = nullptr );

} // end namespace core

} // end namespace viame

#endif // VIAME_CORE_PAIR_STEREO_DETECTIONS_H
