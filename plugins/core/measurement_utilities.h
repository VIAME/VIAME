/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo measurement utility functions
 */

#ifndef VIAME_CORE_MEASUREMENT_UTILITIES_H
#define VIAME_CORE_MEASUREMENT_UTILITIES_H

#include "viame_core_export.h"

#include <vital/types/vector.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image_container.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/detected_object.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/match_set.h>
#include <vital/config/config_block.h>

#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/match_features.h>
#include <vital/algo/estimate_fundamental_matrix.h>
#include <vital/algo/compute_stereo_depth_map.h>

#ifdef VIAME_ENABLE_OPENCV
  #include <opencv2/core/core.hpp>
  #include <opencv2/calib3d/calib3d.hpp>
#endif

#include <string>
#include <vector>
#include <memory>

namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

// =============================================================================
// Free-standing structs and utility functions
// =============================================================================

/// Result structure for full stereo measurement (length + 3D position + error)
struct VIAME_CORE_EXPORT stereo_measurement_result
{
  double length;         // distance between head and tail in 3D
  double x, y, z;        // midpoint 3D position (real-world location)
  double range;          // distance from midpoint to camera
  double rms;            // RMS error
  bool valid;

  stereo_measurement_result()
    : length( 0.0 ), x( 0.0 ), y( 0.0 ), z( 0.0 )
    , range( 0.0 ), rms( 0.0 ), valid( false ) {}
};

/// Add measurement attributes (length, midpoint, range, rms) to a detection
VIAME_CORE_EXPORT void add_measurement_attributes(
  kv::detected_object_sptr det,
  const stereo_measurement_result& measurement );

/// Parse a comma-separated list of matching methods
VIAME_CORE_EXPORT std::vector< std::string > parse_matching_methods(
  const std::string& methods_str );

/// Check if a method requires images
VIAME_CORE_EXPORT bool method_requires_images( const std::string& method );

/// Get list of all valid method names
VIAME_CORE_EXPORT std::vector< std::string > get_valid_methods();

/// Project a point from left camera to right camera using a specified depth
VIAME_CORE_EXPORT kv::vector_2d project_left_to_right(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  double depth );

/// Triangulate a 3D point from stereo correspondences
VIAME_CORE_EXPORT kv::vector_3d triangulate_point(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  const kv::vector_2d& right_point );

/// Compute length between two 3D points from stereo keypoint pairs
VIAME_CORE_EXPORT double compute_stereo_length(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_head,
  const kv::vector_2d& right_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d& right_tail );

/// Compute full stereo measurement including length, 3D position, range, and RMS
VIAME_CORE_EXPORT stereo_measurement_result compute_stereo_measurement(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_head,
  const kv::vector_2d& right_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d& right_tail );

/// Compute a bounding box from keypoints with scale factor
/// If min_aspect_ratio > 0, ensures the smaller dimension is at least
/// min_aspect_ratio times the larger dimension (prevents very thin boxes)
VIAME_CORE_EXPORT kv::bounding_box_d compute_bbox_from_keypoints(
  const kv::vector_2d& head_point,
  const kv::vector_2d& tail_point,
  double box_scale_factor,
  double min_aspect_ratio = 0.10 );

/// Compute epipolar points by sampling depths along a ray from source camera
/// and projecting to target camera. Works on unrectified images.
VIAME_CORE_EXPORT std::vector< kv::vector_2d > compute_epipolar_points(
  const kv::simple_camera_perspective& source_cam,
  const kv::simple_camera_perspective& target_cam,
  const kv::vector_2d& source_point,
  double min_depth, double max_depth, int num_samples );

/// Compute intersection-over-union (IOU) between two bounding boxes
VIAME_CORE_EXPORT double compute_iou(
  const kv::bounding_box_d& bbox1,
  const kv::bounding_box_d& bbox2 );

/// Get the most likely class label from a detection
/// Returns empty string if detection or type is null
VIAME_CORE_EXPORT std::string get_detection_class_label(
  const kv::detected_object_sptr& det );

/// Perform greedy minimum weight assignment given a cost matrix
/// cost_matrix[i][j] is the cost of assigning row i to column j
/// Returns pairs of (row, column) assignments, sorted by increasing cost
/// Ignores costs that are infinity or >= 1e9
VIAME_CORE_EXPORT std::vector< std::pair< int, int > > greedy_assignment(
  const std::vector< std::vector< double > >& cost_matrix,
  int n_rows, int n_cols );

/// Structure to store stereo feature correspondences for head/tail computation
struct VIAME_CORE_EXPORT stereo_feature_correspondence
{
  kv::vector_2d left_point;
  kv::vector_2d right_point;
};

/// Find the two furthest apart points from a set of stereo correspondences
/// Uses the left image points to compute distance
/// Returns true if found (requires at least 2 correspondences), false otherwise
/// Head/tail ordering is consistent: head has smaller x coordinate in left image
VIAME_CORE_EXPORT bool find_furthest_apart_points(
  const std::vector< stereo_feature_correspondence >& correspondences,
  kv::vector_2d& left_head, kv::vector_2d& left_tail,
  kv::vector_2d& right_head, kv::vector_2d& right_tail );

// =============================================================================
// Classes
// =============================================================================

/**
 * \brief Configuration settings for stereo measurement operations
 *
 * This class encapsulates all configuration parameters for stereo measurement
 * and provides standard kwiver get/set configuration functions.
 */
class VIAME_CORE_EXPORT map_keypoints_to_camera_settings
{
public:
  map_keypoints_to_camera_settings();
  ~map_keypoints_to_camera_settings();

  // -------------------------------------------------------------------------
  // Standard kwiver configuration functions
  // -------------------------------------------------------------------------

  /// Get the default configuration block with all parameters
  kv::config_block_sptr get_configuration() const;

  /// Set configuration from a config block
  void set_configuration( kv::config_block_sptr config );

  /// Check if configuration is valid
  bool check_configuration( kv::config_block_sptr config ) const;

  // -------------------------------------------------------------------------
  // Parameter values
  // -------------------------------------------------------------------------

  /// Comma-separated list of matching methods to try in order
  std::string matching_methods;

  /// Default depth (in meters) for depth projection method
  double default_depth;

  /// Template window size (in pixels) for template matching. Must be odd.
  int template_size;

  /// Search range (in pixels) along epipolar line for template matching
  int search_range;

  /// Minimum correlation threshold for template matching (0.0 to 1.0)
  double template_matching_threshold;

  /// Expected disparity (in pixels) for template matching search centering.
  /// If <= 0, disparity is computed from default_depth using camera parameters.
  double template_matching_disparity;

  /// Whether to use SGBM disparity map to estimate initial disparity for template matching.
  /// If enabled and SGBM disparity is available, samples the disparity map near the query
  /// point to get a local disparity estimate, which can be more accurate than using
  /// a fixed default_depth for objects at varying distances.
  bool use_disparity_hint;

  /// Whether to use multi-resolution search for template matching.
  /// If enabled, performs a coarse search first with larger step size, then refines
  /// around the best coarse match. This can improve performance for large search ranges.
  bool use_multires_search;

  /// Step size (in pixels) for coarse search pass in multi-resolution template matching.
  /// Only used when use_multires_search is enabled. Larger values are faster but may
  /// miss optimal matches. Typical values are 2-4.
  int multires_coarse_step;

  /// Whether to use census transform preprocessing for template matching.
  /// Census transform compares each pixel to its neighbors creating a binary pattern,
  /// which is highly robust to illumination changes and camera gain differences.
  bool use_census_transform;

  /// Half-width of the epipolar band for template matching search (in pixels).
  /// Set to 0 for exact epipolar line search (single row).
  /// Set to 1-3 to allow small vertical deviation to handle imperfect rectification.
  /// The search will cover (2 * epipolar_band_halfwidth + 1) rows.
  int epipolar_band_halfwidth;

  /// Minimum depth for epipolar template matching (in camera/calibration units).
  /// Default is 0 (off). Used only when disparity parameters are both 0.
  /// Either disparity or depth parameters must be set for epipolar matching.
  double epipolar_min_depth;

  /// Maximum depth for epipolar template matching (in camera/calibration units).
  /// Default is 0 (off). See epipolar_min_depth.
  double epipolar_max_depth;

  /// Minimum expected disparity in pixels for epipolar template matching.
  /// When both min and max disparity are > 0, they override the depth-based
  /// parameters by converting to depth using the camera intrinsics and baseline:
  ///   depth = focal_length * baseline / disparity
  /// This is unit-independent and often easier to estimate from the images.
  /// Note: min disparity corresponds to max depth (far objects) and vice versa.
  double epipolar_min_disparity;

  /// Maximum expected disparity in pixels for epipolar template matching.
  /// See epipolar_min_disparity for details.
  double epipolar_max_disparity;

  /// Number of sample points along the epipolar line
  int epipolar_num_samples;

  /// Descriptor type for epipolar template matching.
  /// 'ncc' (default): Normalized cross-correlation on grayscale patches.
  /// 'dino': Two-stage DINO + NCC matching using vision transformer features (requires Python).
  std::string epipolar_descriptor_type;

  /// Whether to use distortion coefficients from calibration
  bool use_distortion;

  /// Maximum distance (in pixels) to search for feature matches
  double feature_search_radius;

  /// Inlier threshold for RANSAC fundamental matrix estimation
  double ransac_inlier_scale;

  /// Minimum number of inliers required for valid RANSAC result
  int min_ransac_inliers;

  /// Scale factor to expand bounding box around keypoints
  double box_scale_factor;

  /// Minimum aspect ratio for bounding boxes (smaller dim / larger dim).
  /// Prevents very thin boxes. Set to 0 to disable.
  double box_min_aspect_ratio;

  /// Whether to use depth projection to estimate initial search location for feature matching
  bool use_disparity_aware_feature_search;

  /// Depth to use for disparity-aware feature search (if different from default_depth)
  double feature_search_depth;

  /// Whether to record stereo measurement method as detection attribute
  bool record_stereo_method;

  /// Directory to write debug images showing epipolar search lines.
  /// Empty string (default) disables debug output.
  std::string debug_epipolar_directory;

  /// Detection pairing method: "" (disabled), "epipolar_iou", "keypoint_projection"
  std::string detection_pairing_method;

  /// Threshold for detection pairing: IOU threshold for epipolar_iou (default 0.1),
  /// pixel distance for keypoint_projection (default 50.0)
  double detection_pairing_threshold;

  /// DINO model name when epipolar_descriptor_type is 'dino'.
  /// Supports DINOv3 (e.g., 'dinov3_vits16') and DINOv2 (e.g., 'dinov2_vitb14').
  /// If DINOv3 weights are unavailable, automatically falls back to DINOv2.
  std::string dino3_model_name;

  /// Minimum cosine similarity threshold for DINO matching (0.0 to 1.0).
  /// With top-K + NCC mode (default), this is typically left at 0.
  double dino3_threshold;

  /// Optional path to local DINO weights file. Empty string uses default URL.
  std::string dino3_weights_path;

  /// Number of top DINO candidates to pass to NCC refinement.
  /// The two-stage approach (DINO top-K + NCC) combines DINO's semantic
  /// robustness with NCC's sub-pixel precision. Set to 0 to use DINO-only
  /// matching without NCC refinement.
  int dino3_top_k;

  // -------------------------------------------------------------------------
  // Algorithm pointers (configured via nested algo configuration)
  // -------------------------------------------------------------------------

  /// Feature detector algorithm
  kv::algo::detect_features_sptr feature_detector;

  /// Descriptor extractor algorithm
  kv::algo::extract_descriptors_sptr descriptor_extractor;

  /// Feature matcher algorithm
  kv::algo::match_features_sptr feature_matcher;

  /// Fundamental matrix estimator for RANSAC filtering
  kv::algo::estimate_fundamental_matrix_sptr fundamental_matrix_estimator;

  /// Stereo depth/disparity map algorithm for compute_disparity method
  kv::algo::compute_stereo_depth_map_sptr stereo_depth_map_algorithm;

  // -------------------------------------------------------------------------
  // Utility methods
  // -------------------------------------------------------------------------

  /// Get the parsed list of matching methods
  std::vector< std::string > get_matching_methods() const;

  /// Validate matching methods and return error message (empty if valid)
  std::string validate_matching_methods() const;

  /// Check if any configured method requires images
  bool any_method_requires_images() const;

  /// Check if feature algorithms are properly configured for the specified methods
  std::vector< std::string > check_feature_algorithm_warnings() const;
};


/**
 * \brief Stereo measurement utility class
 *
 * This class provides helper functions for stereo measurement operations
 * including point projection, template matching, SGBM disparity, and
 * feature-based correspondence finding.
 */
class VIAME_CORE_EXPORT map_keypoints_to_camera
{
public:
  map_keypoints_to_camera();
  ~map_keypoints_to_camera();

  // -------------------------------------------------------------------------
  // Configuration from settings
  // -------------------------------------------------------------------------

  /// Configure all parameters from a map_keypoints_to_camera_settings object
  void configure( const map_keypoints_to_camera_settings& settings );

  // -------------------------------------------------------------------------
  // Configuration setters (for individual parameter control)
  // -------------------------------------------------------------------------

  /// Set the default depth for depth projection method
  void set_default_depth( double depth );

  /// Set template matching parameters
  void set_template_params( int template_size, int search_range,
                            double matching_threshold = 0.7,
                            double disparity = 0.0,
                            bool use_sgbm_hint = false,
                            bool use_multires = false,
                            int multires_step = 4,
                            bool use_census = false,
                            int epipolar_band = 0 );

  /// Set epipolar template matching parameters
  void set_epipolar_params( double min_depth, double max_depth, int num_samples );

  /// Set whether to use distortion coefficients
  void set_use_distortion( bool use_distortion );

  /// Set feature matching parameters
  void set_feature_params( double search_radius, double ransac_inlier_scale,
                           int min_ransac_inliers,
                           bool use_disparity_aware_search = false,
                           double feature_search_depth = 5.0 );

  /// Set bounding box scale factor for creating detections from keypoints
  void set_box_scale_factor( double scale_factor );

  /// Set feature detection algorithms
  void set_feature_algorithms(
    kv::algo::detect_features_sptr detector,
    kv::algo::extract_descriptors_sptr extractor,
    kv::algo::match_features_sptr matcher,
    kv::algo::estimate_fundamental_matrix_sptr fundamental_estimator = nullptr );

  // -------------------------------------------------------------------------
  // Core utility functions
  // -------------------------------------------------------------------------

  /// Project a left camera point to the right camera using the default depth
  kv::vector_2d project_left_to_right(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_point ) const;

  /// Compute a bounding box from keypoints with scale factor
  kv::bounding_box_d compute_bbox_from_keypoints(
    const kv::vector_2d& head_point,
    const kv::vector_2d& tail_point ) const;

  // -------------------------------------------------------------------------
  // High-level stereo correspondence functions
  // -------------------------------------------------------------------------

  /// Result structure for stereo correspondence finding
  struct stereo_correspondence_result
  {
    bool success;
    kv::vector_2d left_head;
    kv::vector_2d left_tail;
    kv::vector_2d right_head;
    kv::vector_2d right_tail;
    std::string method_used;
  };

  /// Find stereo correspondences using specified methods in order
  /// Tries each method until one succeeds for both head and tail points
  /// If external_disparity is provided and "external_disparity" method is used,
  /// it will be used to warp points from left to right image.
  stereo_correspondence_result find_stereo_correspondence(
    const std::vector< std::string >& methods,
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_head,
    const kv::vector_2d& left_tail,
    const kv::vector_2d* right_head_input,
    const kv::vector_2d* right_tail_input,
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image,
    const kv::image_container_sptr& external_disparity = nullptr );

#ifdef VIAME_ENABLE_OPENCV
  /// Prepare stereo images for matching (convert to grayscale, rectify, compute disparity)
  struct stereo_image_data
  {
    cv::Mat left_rectified;
    cv::Mat right_rectified;
    cv::Mat disparity_map;
    bool rectified_available;
    bool disparity_available;
  };

  /// Prepare stereo images for the specified methods
  stereo_image_data prepare_stereo_images(
    const std::vector< std::string >& methods,
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image );
#endif

  // -------------------------------------------------------------------------
  // Feature-based matching functions
  // -------------------------------------------------------------------------

  /// Find corresponding point using vital feature detection/matching
  /// Returns true if match found, false otherwise
  /// Updates left_point to the actual feature location if a match is found
  /// If cameras are provided and disparity-aware search is enabled, uses depth
  /// projection to estimate the search location in the right image
  bool find_corresponding_point_feature_descriptor(
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image,
    kv::vector_2d& left_point,
    kv::vector_2d& right_point,
    const kv::simple_camera_perspective* left_cam = nullptr,
    const kv::simple_camera_perspective* right_cam = nullptr );

  /// Find corresponding point using RANSAC feature matching
  /// Returns true if match found, false otherwise
  /// Updates left_point to the actual feature location if a match is found
  /// If cameras are provided and disparity-aware search is enabled, uses depth
  /// projection to estimate the search location in the right image
  bool find_corresponding_point_ransac_feature(
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image,
    kv::vector_2d& left_point,
    kv::vector_2d& right_point,
    const kv::simple_camera_perspective* left_cam = nullptr,
    const kv::simple_camera_perspective* right_cam = nullptr );

  /// Clear cached feature data (call when frame changes)
  void clear_feature_cache();

  /// Set the current frame ID for caching
  void set_frame_id( kv::frame_id_t frame_id );

  /// Get the cached computed disparity map (if available)
  /// This returns the disparity map from the last compute_disparity method call
  kv::image_container_sptr get_cached_disparity() const;

  /// Get the cached rectified left image (if available)
  /// This returns the rectified left image from the last stereo processing call
  kv::image_container_sptr get_cached_rectified_left() const;

  /// Get the cached rectified right image (if available)
  /// This returns the rectified right image from the last stereo processing call
  kv::image_container_sptr get_cached_rectified_right() const;

#ifdef VIAME_ENABLE_OPENCV
  // -------------------------------------------------------------------------
  // OpenCV-based matching functions
  // -------------------------------------------------------------------------

  /// Compute rectification maps for stereo images
  void compute_rectification_maps(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const cv::Size& image_size );

  /// Check if rectification maps have been computed
  bool rectification_computed() const;

  /// Rectify a point from original image space to rectified space
  kv::vector_2d rectify_point(
    const kv::vector_2d& original_point,
    bool is_right_camera ) const;

  /// Unrectify a point from rectified space back to original image space
  kv::vector_2d unrectify_point(
    const kv::vector_2d& rectified_point,
    bool is_right_camera,
    const kv::simple_camera_perspective& camera ) const;

  /// Rectify an image using precomputed maps
  cv::Mat rectify_image( const cv::Mat& image, bool is_right_camera ) const;

  /// Find corresponding point in right image using template matching
  /// Returns true if match found, false otherwise
  /// If disparity_map is provided and use_disparity_hint is enabled,
  /// it will be used to estimate initial disparity for search centering.
  bool find_corresponding_point_template_matching(
    const cv::Mat& left_image_rect,
    const cv::Mat& right_image_rect,
    const kv::vector_2d& left_point_rect,
    kv::vector_2d& right_point_rect,
    const cv::Mat& disparity_map = cv::Mat() ) const;

  /// Find corresponding point by template matching along an arbitrary epipolar line.
  /// Works on unrectified images with epipolar points from any source.
  /// Extracts a template from source_image around source_point and searches
  /// for the best NCC match at each epipolar point in target_image.
  /// Supports census transform via configured settings.
  bool find_corresponding_point_epipolar_template_matching(
    const cv::Mat& source_image,
    const cv::Mat& target_image,
    const kv::vector_2d& source_point,
    const std::vector< kv::vector_2d >& epipolar_points,
    kv::vector_2d& target_point ) const;

  /// Compute SGBM disparity map
  cv::Mat compute_sgbm_disparity(
    const cv::Mat& left_image_rect,
    const cv::Mat& right_image_rect );

  /// Find corresponding point using SGBM disparity
  /// Returns true if valid disparity found, false otherwise
  bool find_corresponding_point_sgbm(
    const cv::Mat& disparity_map,
    const kv::vector_2d& left_point_rect,
    kv::vector_2d& right_point_rect ) const;

  /// Get rectification map for a given camera (for external use)
  const cv::Mat& get_rectification_map_x( bool is_right_camera ) const;
  const cv::Mat& get_rectification_map_y( bool is_right_camera ) const;
#endif

  // -------------------------------------------------------------------------
  // External disparity functions (no OpenCV required)
  // -------------------------------------------------------------------------

  /// Find corresponding point using external disparity map (e.g., from Foundation-Stereo)
  /// The disparity map is expected to be in unrectified image space with disparity
  /// values scaled by 256 (stored as uint16).
  /// Returns true if valid disparity found, false otherwise
  bool find_corresponding_point_external_disparity(
    const kv::image_container_sptr& disparity_image,
    const kv::vector_2d& left_point,
    kv::vector_2d& right_point,
    int search_window = 0 ) const;

private:
  // Configuration
  double m_default_depth;
  int m_template_size;
  int m_search_range;
  double m_template_matching_threshold;
  double m_template_matching_disparity;
  bool m_use_disparity_hint;
  bool m_use_multires_search;
  int m_multires_coarse_step;
  bool m_use_census_transform;
  int m_epipolar_band_halfwidth;
  double m_epipolar_min_depth;
  double m_epipolar_max_depth;
  double m_epipolar_min_disparity;
  double m_epipolar_max_disparity;
  int m_epipolar_num_samples;
  std::string m_epipolar_descriptor_type;
  bool m_use_distortion;
  double m_feature_search_radius;
  double m_ransac_inlier_scale;
  int m_min_ransac_inliers;
  double m_box_scale_factor;
  double m_box_min_aspect_ratio;
  bool m_use_disparity_aware_feature_search;
  double m_feature_search_depth;
  std::string m_debug_epipolar_directory;
  unsigned m_debug_frame_counter;

  // DINO matching settings
  std::string m_dino3_model_name;
  double m_dino3_threshold;
  std::string m_dino3_weights_path;
  int m_dino3_top_k;

  // Feature algorithms
  kv::algo::detect_features_sptr m_feature_detector;
  kv::algo::extract_descriptors_sptr m_descriptor_extractor;
  kv::algo::match_features_sptr m_feature_matcher;
  kv::algo::estimate_fundamental_matrix_sptr m_fundamental_matrix_estimator;

  // Stereo depth map algorithm for compute_disparity method
  kv::algo::compute_stereo_depth_map_sptr m_stereo_depth_map_algorithm;

  // Cached computed disparity from the algorithm
  kv::image_container_sptr m_cached_compute_disparity;

  // Cached feature detection/descriptor results per frame
  kv::feature_set_sptr m_cached_left_features;
  kv::feature_set_sptr m_cached_right_features;
  kv::descriptor_set_sptr m_cached_left_descriptors;
  kv::descriptor_set_sptr m_cached_right_descriptors;
  kv::match_set_sptr m_cached_matches;
  kv::frame_id_t m_cached_frame_id;

#ifdef VIAME_ENABLE_OPENCV
  // Rectification maps
  bool m_rectification_computed;
  cv::Mat m_rectification_map_left_x;
  cv::Mat m_rectification_map_left_y;
  cv::Mat m_rectification_map_right_x;
  cv::Mat m_rectification_map_right_y;

  // Rectification matrices for unrectifying points
  cv::Mat m_K1, m_K2, m_R1, m_R2, m_P1, m_P2, m_D1, m_D2;

  // Template matching helpers
  struct prepared_template
  {
    cv::Mat ncc_template;
    cv::Mat census_template;
    bool valid;
    prepared_template() : valid( false ) {}
  };

  /// Prepare a source template for matching (bounds check + extraction + census)
  /// Returns false if template can't be extracted (point too close to edge)
  bool prepare_source_template(
    const cv::Mat& source_image, int x, int y,
    prepared_template& tmpl ) const;

  /// Score a candidate point against a prepared template
  /// Returns NCC-like score (higher is better, 0-1 range)
  /// Returns -1.0 if the candidate point is too close to the image edge
  double score_template_at_point(
    const prepared_template& tmpl,
    const cv::Mat& target_image, int x, int y ) const;

  // Cached stereo image data
  stereo_image_data m_cached_stereo_images;
#endif
};

} // end namespace core

} // end namespace viame

#endif // VIAME_CORE_MEASUREMENT_UTILITIES_H
