/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo detection pairing process implementation
 */

#include <algorithm>
#include <limits>
#include <cmath>
#include <map>
#include <set>

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/object_track_set.h>
#include <vital/types/bounding_box.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/image.h>
#include <vital/types/image_container.h>
#include <vital/types/feature.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/match_set.h>

#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/match_features.h>
#include <vital/algo/estimate_homography.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

#include "pair_stereo_detections_process.h"
#include "pair_stereo_detections.h"
#include "pair_stereo_tracks.h"
#include "measurement_utilities.h"
#include "camera_rig_io.h"

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

// Config traits
create_config_trait( matching_method, std::string, "iou",
  "Matching method to use. Options: 'iou' (bounding box overlap), "
  "'calibration' (uses stereo geometry and reprojection error), "
  "'feature_matching' (uses feature detection and matching within bounding boxes)" );

create_config_trait( calibration_file, std::string, "",
  "Stereo calibration file (JSON format). Required when matching_method is 'calibration'." );

create_config_trait( iou_threshold, double, "0.1",
  "Minimum IOU (Intersection over Union) threshold for matching detections. "
  "Used with 'iou' method. Pairs with IOU below this value will not be matched." );

create_config_trait( max_reprojection_error, double, "10.0",
  "Maximum reprojection error (in pixels) for valid matches. "
  "Used with 'calibration' method." );

create_config_trait( default_depth, double, "0",
  "Default depth for projecting points between cameras. Units must match "
  "the calibration T vector (e.g. millimeters if T is in mm). When set to "
  "0 (default), keypoint_projection uses depth-independent epipolar line "
  "distance instead, which requires no depth prior." );

create_config_trait( require_class_match, bool, "true",
  "If true, only detections with the same class label can be matched." );

create_config_trait( use_optimal_assignment, bool, "true",
  "If true, use optimal (greedy) assignment to maximize matching quality. "
  "If false, use simple greedy matching in order of left detections." );


// Feature matching config traits
create_config_trait( min_feature_match_count, int, "5",
  "Minimum number of feature matches required between two detection boxes "
  "to consider them a valid match. Used with 'feature_matching' method." );

create_config_trait( min_feature_match_ratio, double, "0.1",
  "Minimum ratio of feature matches to total features detected in the left box. "
  "A ratio of 0.1 means at least 10% of left features must match. "
  "Used with 'feature_matching' method." );

create_config_trait( use_homography_filtering, bool, "true",
  "If true, estimate a homography between matched features and reject outliers. "
  "This helps filter spurious matches that don't follow a consistent geometric "
  "transformation. Used with 'feature_matching' method." );

create_config_trait( homography_inlier_threshold, double, "5.0",
  "Maximum reprojection error (in pixels) for a match to be considered an inlier "
  "when estimating the homography. Used with 'feature_matching' method." );

create_config_trait( min_homography_inlier_ratio, double, "0.5",
  "Minimum ratio of inlier matches to total matches after homography estimation. "
  "A ratio of 0.5 means at least 50% of matches must be inliers. "
  "Used with 'feature_matching' method." );

create_config_trait( box_expansion_factor, double, "1.1",
  "Factor to expand bounding boxes when extracting features. "
  "A value of 1.1 expands boxes by 10%. Used with 'feature_matching' method." );

create_config_trait( compute_head_tail_points, bool, "false",
  "If true, compute head and tail keypoints from the two furthest apart inlier "
  "feature matches and add them to the paired detections. Only applies when "
  "matching_method is 'feature_matching'. The head/tail points are useful for "
  "downstream stereo measurement algorithms." );

create_config_trait( min_inliers_for_head_tail, int, "4",
  "Minimum number of inlier feature matches required to compute head/tail points. "
  "If fewer inliers are found, no head/tail points will be added to the detection." );

// Epipolar IOU config traits
create_config_trait( epipolar_iou_threshold, double, "0.1",
  "Minimum IOU threshold after projecting left bounding box to right image "
  "using camera geometry. Used with 'epipolar_iou' method." );

// Keypoint projection config traits
create_config_trait( max_keypoint_distance, double, "50.0",
  "Maximum average pixel distance between projected left keypoints and right "
  "detection keypoints. Used with 'keypoint_projection' method." );


// Port traits
create_port_trait( detected_object_set1, detected_object_set,
  "Detections from camera 1 (left)" );
create_port_trait( detected_object_set2, detected_object_set,
  "Detections from camera 2 (right)" );
create_port_trait( object_track_set1, object_track_set,
  "Output tracks for camera 1" );
create_port_trait( object_track_set2, object_track_set,
  "Output tracks for camera 2" );
create_port_trait( image1, image,
  "Image from camera 1 (left). Required for feature_matching method." );
create_port_trait( image2, image,
  "Image from camera 2 (right). Required for feature_matching method." );

// =============================================================================
// Private implementation class
class pair_stereo_detections_process::priv
{
public:
  explicit priv( pair_stereo_detections_process* parent );
  ~priv();

  // Build option structs from configuration
  iou_matching_options get_iou_options() const;
  calibration_matching_options get_calibration_options() const;
  feature_matching_options get_feature_options() const;
  feature_matching_algorithms get_feature_algorithms() const;
  epipolar_iou_matching_options get_epipolar_iou_options() const;
  keypoint_projection_matching_options get_keypoint_projection_options() const;

  // Configuration values
  std::string m_matching_method;
  std::string m_calibration_file;
  double m_iou_threshold;
  double m_max_reprojection_error;
  double m_default_depth;
  bool m_require_class_match;
  bool m_use_optimal_assignment;

  // Feature matching configuration
  int m_min_feature_match_count;
  double m_min_feature_match_ratio;
  bool m_use_homography_filtering;
  double m_homography_inlier_threshold;
  double m_min_homography_inlier_ratio;
  double m_box_expansion_factor;
  bool m_compute_head_tail_points;
  int m_min_inliers_for_head_tail;

  // Epipolar IOU / keypoint projection configuration
  double m_epipolar_iou_threshold;
  double m_max_keypoint_distance;

  // Calibration data
  kv::camera_rig_stereo_sptr m_calibration;

  // Feature matching algorithms
  kv::algo::detect_features_sptr m_feature_detector;
  kv::algo::extract_descriptors_sptr m_descriptor_extractor;
  kv::algo::match_features_sptr m_feature_matcher;
  kv::algo::estimate_homography_sptr m_homography_estimator;

  // Track pairing (shared with measure_objects_process)
  stereo_track_pairer m_track_pairer;

  pair_stereo_detections_process* parent;
};

// -----------------------------------------------------------------------------
pair_stereo_detections_process::priv
::priv( pair_stereo_detections_process* ptr )
  : m_matching_method( "iou" )
  , m_calibration_file( "" )
  , m_iou_threshold( 0.1 )
  , m_max_reprojection_error( 10.0 )
  , m_default_depth( 0.0 )
  , m_require_class_match( true )
  , m_use_optimal_assignment( true )
  , m_min_feature_match_count( 5 )
  , m_min_feature_match_ratio( 0.1 )
  , m_use_homography_filtering( true )
  , m_homography_inlier_threshold( 5.0 )
  , m_min_homography_inlier_ratio( 0.5 )
  , m_box_expansion_factor( 1.1 )
  , m_compute_head_tail_points( false )
  , m_min_inliers_for_head_tail( 4 )
  , m_epipolar_iou_threshold( 0.1 )
  , m_max_keypoint_distance( 50.0 )
  , parent( ptr )
{
}

// -----------------------------------------------------------------------------
pair_stereo_detections_process::priv
::~priv()
{
}

// -----------------------------------------------------------------------------
iou_matching_options
pair_stereo_detections_process::priv
::get_iou_options() const
{
  iou_matching_options options;
  options.iou_threshold = m_iou_threshold;
  options.require_class_match = m_require_class_match;
  options.use_optimal_assignment = m_use_optimal_assignment;
  return options;
}

// -----------------------------------------------------------------------------
calibration_matching_options
pair_stereo_detections_process::priv
::get_calibration_options() const
{
  calibration_matching_options options;
  options.max_reprojection_error = m_max_reprojection_error;
  options.default_depth = m_default_depth;
  options.require_class_match = m_require_class_match;
  options.use_optimal_assignment = m_use_optimal_assignment;
  return options;
}

// -----------------------------------------------------------------------------
feature_matching_options
pair_stereo_detections_process::priv
::get_feature_options() const
{
  feature_matching_options options;
  options.min_feature_match_count = m_min_feature_match_count;
  options.min_feature_match_ratio = m_min_feature_match_ratio;
  options.use_homography_filtering = m_use_homography_filtering;
  options.homography_inlier_threshold = m_homography_inlier_threshold;
  options.min_homography_inlier_ratio = m_min_homography_inlier_ratio;
  options.box_expansion_factor = m_box_expansion_factor;
  options.require_class_match = m_require_class_match;
  options.use_optimal_assignment = m_use_optimal_assignment;
  return options;
}

// -----------------------------------------------------------------------------
feature_matching_algorithms
pair_stereo_detections_process::priv
::get_feature_algorithms() const
{
  feature_matching_algorithms algorithms;
  algorithms.feature_detector = m_feature_detector;
  algorithms.descriptor_extractor = m_descriptor_extractor;
  algorithms.feature_matcher = m_feature_matcher;
  algorithms.homography_estimator = m_homography_estimator;
  return algorithms;
}

// -----------------------------------------------------------------------------
epipolar_iou_matching_options
pair_stereo_detections_process::priv
::get_epipolar_iou_options() const
{
  epipolar_iou_matching_options options;
  options.iou_threshold = m_epipolar_iou_threshold;
  options.default_depth = m_default_depth;
  options.require_class_match = m_require_class_match;
  options.use_optimal_assignment = m_use_optimal_assignment;
  return options;
}

// -----------------------------------------------------------------------------
keypoint_projection_matching_options
pair_stereo_detections_process::priv
::get_keypoint_projection_options() const
{
  keypoint_projection_matching_options options;
  options.max_keypoint_distance = m_max_keypoint_distance;
  options.default_depth = m_default_depth;
  options.require_class_match = m_require_class_match;
  options.use_optimal_assignment = m_use_optimal_assignment;
  return options;
}

// =============================================================================
pair_stereo_detections_process
::pair_stereo_detections_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new pair_stereo_detections_process::priv( this ) )
{
  make_ports();
  make_config();
}

pair_stereo_detections_process
::~pair_stereo_detections_process()
{
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process
::make_ports()
{
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // Input ports - either detections or tracks can be connected
  declare_input_port_using_trait( detected_object_set1, optional );
  declare_input_port_using_trait( detected_object_set2, optional );
  declare_input_port_using_trait( object_track_set1, optional );
  declare_input_port_using_trait( object_track_set2, optional );
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image1, optional );
  declare_input_port_using_trait( image2, optional );

  // Output ports
  declare_output_port_using_trait( object_track_set1, optional );
  declare_output_port_using_trait( object_track_set2, optional );
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process
::make_config()
{
  declare_config_using_trait( matching_method );
  declare_config_using_trait( calibration_file );
  declare_config_using_trait( iou_threshold );
  declare_config_using_trait( max_reprojection_error );
  declare_config_using_trait( default_depth );
  declare_config_using_trait( require_class_match );
  declare_config_using_trait( use_optimal_assignment );

  // Feature matching configuration
  declare_config_using_trait( min_feature_match_count );
  declare_config_using_trait( min_feature_match_ratio );
  declare_config_using_trait( use_homography_filtering );
  declare_config_using_trait( homography_inlier_threshold );
  declare_config_using_trait( min_homography_inlier_ratio );
  declare_config_using_trait( box_expansion_factor );
  declare_config_using_trait( compute_head_tail_points );
  declare_config_using_trait( min_inliers_for_head_tail );

  // Epipolar IOU / keypoint projection configuration
  declare_config_using_trait( epipolar_iou_threshold );
  declare_config_using_trait( max_keypoint_distance );

  // Track pairing configuration (shared with measure_objects_process)
  {
    kv::config_block_sptr tp_config = d->m_track_pairer.get_configuration();
    for( auto const& key : tp_config->available_values() )
    {
      declare_configuration_key( key,
        tp_config->get_value< std::string >( key ),
        tp_config->get_description( key ) );
    }
  }

  // Algorithm configuration (nested algorithms for feature matching)
  kv::algo::detect_features::get_nested_algo_configuration(
    "feature_detector", get_config(), d->m_feature_detector );

  kv::algo::extract_descriptors::get_nested_algo_configuration(
    "descriptor_extractor", get_config(), d->m_descriptor_extractor );

  kv::algo::match_features::get_nested_algo_configuration(
    "feature_matcher", get_config(), d->m_feature_matcher );

  kv::algo::estimate_homography::get_nested_algo_configuration(
    "homography_estimator", get_config(), d->m_homography_estimator );
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process
::_configure()
{
  d->m_matching_method = config_value_using_trait( matching_method );
  d->m_calibration_file = config_value_using_trait( calibration_file );
  d->m_iou_threshold = config_value_using_trait( iou_threshold );
  d->m_max_reprojection_error = config_value_using_trait( max_reprojection_error );
  d->m_default_depth = config_value_using_trait( default_depth );
  d->m_require_class_match = config_value_using_trait( require_class_match );
  d->m_use_optimal_assignment = config_value_using_trait( use_optimal_assignment );

  // Feature matching configuration
  d->m_min_feature_match_count = config_value_using_trait( min_feature_match_count );
  d->m_min_feature_match_ratio = config_value_using_trait( min_feature_match_ratio );
  d->m_use_homography_filtering = config_value_using_trait( use_homography_filtering );
  d->m_homography_inlier_threshold = config_value_using_trait( homography_inlier_threshold );
  d->m_min_homography_inlier_ratio = config_value_using_trait( min_homography_inlier_ratio );
  d->m_box_expansion_factor = config_value_using_trait( box_expansion_factor );
  d->m_compute_head_tail_points = config_value_using_trait( compute_head_tail_points );
  d->m_min_inliers_for_head_tail = config_value_using_trait( min_inliers_for_head_tail );

  // Epipolar IOU / keypoint projection configuration
  d->m_epipolar_iou_threshold = config_value_using_trait( epipolar_iou_threshold );
  d->m_max_keypoint_distance = config_value_using_trait( max_keypoint_distance );

  // Track pairing configuration
  d->m_track_pairer.set_configuration( get_config() );

  // Validate matching method
  if( d->m_matching_method != "iou" &&
      d->m_matching_method != "calibration" &&
      d->m_matching_method != "feature_matching" &&
      d->m_matching_method != "epipolar_iou" &&
      d->m_matching_method != "keypoint_projection" )
  {
    throw std::runtime_error( "Invalid matching_method: '" + d->m_matching_method +
                              "'. Must be 'iou', 'calibration', 'feature_matching', "
                              "'epipolar_iou', or 'keypoint_projection'." );
  }

  // Load calibration if using a calibration-based method
  bool need_calibration = ( d->m_matching_method == "calibration" ) ||
                           ( d->m_matching_method == "epipolar_iou" ) ||
                           ( d->m_matching_method == "keypoint_projection" );

  if( need_calibration )
  {
    if( d->m_calibration_file.empty() )
    {
      throw std::runtime_error( "calibration_file is required when matching_method is '" +
                                d->m_matching_method + "'" );
    }

    d->m_calibration = viame::read_stereo_rig( d->m_calibration_file );

    if( !d->m_calibration || !d->m_calibration->left() || !d->m_calibration->right() )
    {
      throw std::runtime_error( "Failed to load calibration file: " + d->m_calibration_file );
    }

    LOG_INFO( logger(), "Loaded stereo calibration from: " << d->m_calibration_file );
  }

  // Configure feature matching algorithms if using feature_matching method or compute_head_tail_points
  bool need_feature_algorithms = ( d->m_matching_method == "feature_matching" ) ||
                                  d->m_compute_head_tail_points;

  if( need_feature_algorithms )
  {
    // Get nested algorithm configuration
    kv::config_block_sptr config = get_config();

    kv::algo::detect_features::set_nested_algo_configuration(
      "feature_detector", config, d->m_feature_detector );

    kv::algo::extract_descriptors::set_nested_algo_configuration(
      "descriptor_extractor", config, d->m_descriptor_extractor );

    kv::algo::match_features::set_nested_algo_configuration(
      "feature_matcher", config, d->m_feature_matcher );

    if( d->m_use_homography_filtering )
    {
      kv::algo::estimate_homography::set_nested_algo_configuration(
        "homography_estimator", config, d->m_homography_estimator );
    }

    // Validate that required algorithms are configured
    std::string context = d->m_matching_method == "feature_matching"
      ? "matching_method is 'feature_matching'"
      : "compute_head_tail_points is enabled";

    if( !d->m_feature_detector )
    {
      throw std::runtime_error(
        "feature_detector algorithm is required when " + context + ". "
        "Configure it using 'feature_detector:type = <algorithm_name>'" );
    }

    if( !d->m_descriptor_extractor )
    {
      throw std::runtime_error(
        "descriptor_extractor algorithm is required when " + context + ". "
        "Configure it using 'descriptor_extractor:type = <algorithm_name>'" );
    }

    if( !d->m_feature_matcher )
    {
      throw std::runtime_error(
        "feature_matcher algorithm is required when " + context + ". "
        "Configure it using 'feature_matcher:type = <algorithm_name>'" );
    }

    if( d->m_use_homography_filtering && !d->m_homography_estimator )
    {
      throw std::runtime_error(
        "homography_estimator algorithm is required when use_homography_filtering is true. "
        "Configure it using 'homography_estimator:type = <algorithm_name>'" );
    }

    LOG_INFO( logger(), "Feature matching algorithms configured" );
  }

  LOG_INFO( logger(), "Stereo detection pairing configured:" );
  LOG_INFO( logger(), "  Matching method: " << d->m_matching_method );
  if( d->m_matching_method == "iou" )
  {
    LOG_INFO( logger(), "  IOU threshold: " << d->m_iou_threshold );
  }
  else if( d->m_matching_method == "calibration" )
  {
    LOG_INFO( logger(), "  Max reprojection error: " << d->m_max_reprojection_error );
    LOG_INFO( logger(), "  Default depth: " << d->m_default_depth );
  }
  else if( d->m_matching_method == "feature_matching" )
  {
    LOG_INFO( logger(), "  Min feature match count: " << d->m_min_feature_match_count );
    LOG_INFO( logger(), "  Min feature match ratio: " << d->m_min_feature_match_ratio );
    LOG_INFO( logger(), "  Use homography filtering: " << ( d->m_use_homography_filtering ? "true" : "false" ) );
    if( d->m_use_homography_filtering )
    {
      LOG_INFO( logger(), "  Homography inlier threshold: " << d->m_homography_inlier_threshold );
      LOG_INFO( logger(), "  Min homography inlier ratio: " << d->m_min_homography_inlier_ratio );
    }
    LOG_INFO( logger(), "  Box expansion factor: " << d->m_box_expansion_factor );
  }
  else if( d->m_matching_method == "epipolar_iou" )
  {
    LOG_INFO( logger(), "  Epipolar IOU threshold: " << d->m_epipolar_iou_threshold );
    LOG_INFO( logger(), "  Default depth: " << d->m_default_depth );
  }
  else if( d->m_matching_method == "keypoint_projection" )
  {
    LOG_INFO( logger(), "  Max keypoint distance: " << d->m_max_keypoint_distance );
    if( d->m_default_depth > 0 )
    {
      LOG_INFO( logger(), "  Default depth: " << d->m_default_depth << " (projection mode)" );
    }
    else
    {
      LOG_INFO( logger(), "  Default depth: 0 (epipolar line distance mode)" );
    }
  }
  LOG_INFO( logger(), "  Require class match: " << ( d->m_require_class_match ? "true" : "false" ) );
  LOG_INFO( logger(), "  Use optimal assignment: " << ( d->m_use_optimal_assignment ? "true" : "false" ) );
  LOG_INFO( logger(), "  Output unmatched: " << ( d->m_track_pairer.output_unmatched() ? "true" : "false" ) );
  LOG_INFO( logger(), "  Compute head/tail points: " << ( d->m_compute_head_tail_points ? "true" : "false" ) );
  if( d->m_compute_head_tail_points )
  {
    LOG_INFO( logger(), "  Min inliers for head/tail: " << d->m_min_inliers_for_head_tail );
  }
  LOG_INFO( logger(), "  Accumulate track pairings: " << ( d->m_track_pairer.accumulation_enabled() ? "true" : "false" ) );
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process
::_step()
{
  // Check for end-of-stream on any connected input port
  {
    auto ts_peek = peek_at_port_using_trait( timestamp );
    bool is_complete = ( ts_peek.datum->type() == sprokit::datum::complete );

    if( !is_complete && has_input_port_edge_using_trait( object_track_set1 ) )
    {
      auto peek = peek_at_port_using_trait( object_track_set1 );
      is_complete = ( peek.datum->type() == sprokit::datum::complete );
    }

    if( !is_complete && has_input_port_edge_using_trait( object_track_set2 ) )
    {
      auto peek = peek_at_port_using_trait( object_track_set2 );
      is_complete = ( peek.datum->type() == sprokit::datum::complete );
    }

    if( !is_complete && has_input_port_edge_using_trait( detected_object_set1 ) )
    {
      auto peek = peek_at_port_using_trait( detected_object_set1 );
      is_complete = ( peek.datum->type() == sprokit::datum::complete );
    }

    if( !is_complete && has_input_port_edge_using_trait( detected_object_set2 ) )
    {
      auto peek = peek_at_port_using_trait( detected_object_set2 );
      is_complete = ( peek.datum->type() == sprokit::datum::complete );
    }

    if( is_complete )
    {
      mark_process_as_complete();
      auto cd = sprokit::datum::complete_datum();
      push_datum_to_port_using_trait( object_track_set1, cd );
      push_datum_to_port_using_trait( object_track_set2, cd );
      return;
    }
  }

  // Grab timestamp (always required)
  auto timestamp = grab_from_port_using_trait( timestamp );

  // Determine input source and grab detections + track IDs
  std::vector< kv::detected_object_sptr > detections1, detections2;
  std::vector< kv::track_id_t > track_ids1, track_ids2;
  kv::object_track_set_sptr saved_track_set1, saved_track_set2;

  bool use_detections1 = has_input_port_edge_using_trait( detected_object_set1 );
  bool use_detections2 = has_input_port_edge_using_trait( detected_object_set2 );
  bool use_tracks1 = has_input_port_edge_using_trait( object_track_set1 );
  bool use_tracks2 = has_input_port_edge_using_trait( object_track_set2 );

  // Validate input configuration
  if( !use_detections1 && !use_tracks1 )
  {
    throw std::runtime_error( "No input connected for camera 1. "
      "Connect either detected_object_set1 or object_track_set1." );
  }
  if( !use_detections2 && !use_tracks2 )
  {
    throw std::runtime_error( "No input connected for camera 2. "
      "Connect either detected_object_set2 or object_track_set2." );
  }

  // Grab camera 1 input
  if( use_detections1 )
  {
    auto detection_set1 = grab_from_port_using_trait( detected_object_set1 );
    kv::track_id_t synthetic_id = 0;
    for( const auto& det : *detection_set1 )
    {
      detections1.push_back( det );
      track_ids1.push_back( synthetic_id++ );
    }
  }
  else if( use_tracks1 )
  {
    saved_track_set1 = grab_from_port_using_trait( object_track_set1 );
    for( const auto& track : saved_track_set1->tracks() )
    {
      // Get the state for the current frame
      auto it = track->find( timestamp.get_frame() );
      if( it != track->end() )
      {
        auto state = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( state && state->detection() )
        {
          detections1.push_back( state->detection() );
          track_ids1.push_back( track->id() );
        }
      }
    }
  }

  // Grab camera 2 input
  if( use_detections2 )
  {
    auto detection_set2 = grab_from_port_using_trait( detected_object_set2 );
    kv::track_id_t synthetic_id = 0;
    for( const auto& det : *detection_set2 )
    {
      detections2.push_back( det );
      track_ids2.push_back( synthetic_id++ );
    }
  }
  else if( use_tracks2 )
  {
    saved_track_set2 = grab_from_port_using_trait( object_track_set2 );
    for( const auto& track : saved_track_set2->tracks() )
    {
      // Get the state for the current frame
      auto it = track->find( timestamp.get_frame() );
      if( it != track->end() )
      {
        auto state = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( state && state->detection() )
        {
          detections2.push_back( state->detection() );
          track_ids2.push_back( track->id() );
        }
      }
    }
  }

  // Grab images if needed for feature matching or head/tail point computation
  kv::image_container_sptr image1, image2;
  bool need_images = ( d->m_matching_method == "feature_matching" ) ||
                     d->m_compute_head_tail_points;

  if( need_images )
  {
    bool has_image1 = has_input_port_edge_using_trait( image1 );
    bool has_image2 = has_input_port_edge_using_trait( image2 );

    if( !has_image1 || !has_image2 )
    {
      if( d->m_matching_method == "feature_matching" )
      {
        throw std::runtime_error( "Images are required for feature_matching method. "
          "Connect image1 and image2 ports." );
      }
      else if( d->m_compute_head_tail_points )
      {
        LOG_WARN( logger(), "Images not connected but compute_head_tail_points is enabled. "
          "Head/tail points will not be computed." );
        // Reset flag since we can't compute without images
        need_images = false;
      }
    }
    else
    {
      image1 = grab_from_port_using_trait( image1 );
      image2 = grab_from_port_using_trait( image2 );
    }
  }

  // Find matches using configured method via shared dispatch
  detection_pairing_params dp;
  dp.method = d->m_matching_method;
  dp.require_class_match = d->m_require_class_match;
  dp.use_optimal_assignment = d->m_use_optimal_assignment;
  dp.default_depth = d->m_default_depth;

  // Set threshold based on method
  if( dp.method == "iou" )
    dp.threshold = d->m_iou_threshold;
  else if( dp.method == "epipolar_iou" )
    dp.threshold = d->m_epipolar_iou_threshold;
  else if( dp.method == "calibration" )
    dp.threshold = d->m_max_reprojection_error;
  else if( dp.method == "keypoint_projection" )
    dp.threshold = d->m_max_keypoint_distance;

  // Resolve cameras if needed
  const kv::simple_camera_perspective* left_cam_ptr = nullptr;
  const kv::simple_camera_perspective* right_cam_ptr = nullptr;

  if( d->m_calibration && d->m_calibration->left() && d->m_calibration->right() )
  {
    left_cam_ptr = dynamic_cast< const kv::simple_camera_perspective* >(
      d->m_calibration->left().get() );
    right_cam_ptr = dynamic_cast< const kv::simple_camera_perspective* >(
      d->m_calibration->right().get() );
  }

  auto feat_algos = d->get_feature_algorithms();
  auto feat_opts = d->get_feature_options();

  std::vector< std::pair< int, int > > matches = find_stereo_detection_matches(
    dp, detections1, detections2,
    left_cam_ptr, right_cam_ptr, image1, image2,
    &feat_algos, &feat_opts, logger() );

  LOG_DEBUG( logger(), "Frame " << timestamp.get_frame()
             << ": Found " << matches.size() << " matches out of "
             << detections1.size() << " left, " << detections2.size() << " right detections" );

  // Compute head/tail keypoints if enabled and images are available
  if( d->m_compute_head_tail_points && image1 && image2 )
  {
    for( const auto& match : matches )
    {
      int i1 = match.first;
      int i2 = match.second;

      // Get feature correspondences with outlier rejection
      auto correspondences = compute_detection_feature_correspondences(
        detections1[i1], detections2[i2], image1, image2,
        d->get_feature_algorithms(), d->get_feature_options(), logger() );

      if( static_cast< int >( correspondences.size() ) >= d->m_min_inliers_for_head_tail )
      {
        kv::vector_2d left_head, left_tail, right_head, right_tail;

        if( find_furthest_apart_points( correspondences,
                                         left_head, left_tail,
                                         right_head, right_tail ) )
        {
          detections1[i1]->add_keypoint( "head", kv::point_2d( left_head.x(), left_head.y() ) );
          detections1[i1]->add_keypoint( "tail", kv::point_2d( left_tail.x(), left_tail.y() ) );
          detections2[i2]->add_keypoint( "head", kv::point_2d( right_head.x(), right_head.y() ) );
          detections2[i2]->add_keypoint( "tail", kv::point_2d( right_tail.x(), right_tail.y() ) );

          LOG_DEBUG( logger(), "Added head/tail keypoints for matched pair with "
                     << correspondences.size() << " inlier correspondences" );
        }
      }
      else
      {
        LOG_DEBUG( logger(), "Not enough inliers (" << correspondences.size()
                   << " < " << d->m_min_inliers_for_head_tail
                   << ") to compute head/tail keypoints" );
      }
    }
  }

  // Branch: accumulation mode vs per-frame mode
  if( d->m_track_pairer.accumulation_enabled() )
  {
    // Accumulate this frame's pairings
    d->m_track_pairer.accumulate_frame_pairings( matches, detections1, detections2,
                                                 track_ids1, track_ids2, timestamp );

    // Check if input stream is complete
    auto port_info = peek_at_port_using_trait( timestamp );
    bool is_input_complete = port_info.datum->type() == sprokit::datum::complete;

    if( is_input_complete )
    {
      // Resolve accumulated pairings and push final results
      std::vector< kv::track_sptr > output_trks1, output_trks2;
      d->m_track_pairer.resolve_accumulated_pairings( output_trks1, output_trks2 );

      auto output_track_set1 = std::make_shared< kv::object_track_set >( output_trks1 );
      auto output_track_set2 = std::make_shared< kv::object_track_set >( output_trks2 );

      push_to_port_using_trait( object_track_set1, output_track_set1 );
      push_to_port_using_trait( object_track_set2, output_track_set2 );

      LOG_INFO( logger(), "Resolved " << output_trks1.size() << " left and "
                << output_trks2.size() << " right accumulated tracks" );

      // Send complete datum and mark process as complete
      auto complete_dat = sprokit::datum::complete_datum();
      push_datum_to_port_using_trait( object_track_set1, complete_dat );
      push_datum_to_port_using_trait( object_track_set2, complete_dat );
      mark_process_as_complete();
    }
    else
    {
      // Push empty datum while accumulating
      auto empty_dat = sprokit::datum::empty_datum();
      push_datum_to_port_using_trait( object_track_set1, empty_dat );
      push_datum_to_port_using_trait( object_track_set2, empty_dat );
    }
  }
  else if( saved_track_set1 && saved_track_set2 )
  {
    // Per-frame mode with track pass-through: delegate to shared track pairer
    std::vector< kv::track_sptr > remapped_left, remapped_right;
    d->m_track_pairer.remap_tracks_per_frame(
      saved_track_set1, saved_track_set2, matches,
      track_ids1, track_ids2, remapped_left, remapped_right );

    auto output_track_set1 = std::make_shared< kv::object_track_set >( remapped_left );
    auto output_track_set2 = std::make_shared< kv::object_track_set >( remapped_right );

    push_to_port_using_trait( object_track_set1, output_track_set1 );
    push_to_port_using_trait( object_track_set2, output_track_set2 );
  }
  else
  {
    // Per-frame mode with detection inputs: create single-frame tracks
    std::vector< bool > has_match1( detections1.size(), false );
    std::vector< bool > has_match2( detections2.size(), false );

    std::vector< kv::track_sptr > output_trks1, output_trks2;

    for( const auto& match : matches )
    {
      int i1 = match.first;
      int i2 = match.second;

      has_match1[i1] = true;
      has_match2[i2] = true;

      // Average class labels for matched pair if enabled
      if( d->m_track_pairer.average_stereo_classes() )
      {
        bool weighted = d->m_track_pairer.use_weighted_averaging();
        bool sbc = d->m_track_pairer.use_scaled_by_conf();
        std::string ign = d->m_track_pairer.class_averaging_ignore_class();
        std::vector< kv::detected_object_sptr > left_dets = { detections1[i1] };
        std::vector< kv::detected_object_sptr > right_dets = { detections2[i2] };
        auto avg_dot = compute_stereo_average_classification(
          left_dets, right_dets, weighted, sbc, ign );
        if( avg_dot )
        {
          detections1[i1]->set_type( avg_dot );
          detections2[i2]->set_type( avg_dot );
        }
      }

      auto state1 = std::make_shared< kv::object_track_state >( timestamp, detections1[i1] );
      auto state2 = std::make_shared< kv::object_track_state >( timestamp, detections2[i2] );

      auto track1 = kv::track::create();
      kv::track_id_t tid = d->m_track_pairer.allocate_track_id();
      track1->set_id( tid );
      track1->append( state1 );

      auto track2 = kv::track::create();
      track2->set_id( tid );
      track2->append( state2 );

      output_trks1.push_back( track1 );
      output_trks2.push_back( track2 );
    }

    // Add unmatched detections as separate tracks if configured
    if( d->m_track_pairer.output_unmatched() )
    {
      for( size_t i = 0; i < detections1.size(); ++i )
      {
        if( !has_match1[i] )
        {
          auto state = std::make_shared< kv::object_track_state >( timestamp, detections1[i] );
          auto track = kv::track::create();
          track->set_id( d->m_track_pairer.allocate_track_id() );
          track->append( state );
          output_trks1.push_back( track );
        }
      }

      for( size_t i = 0; i < detections2.size(); ++i )
      {
        if( !has_match2[i] )
        {
          auto state = std::make_shared< kv::object_track_state >( timestamp, detections2[i] );
          auto track = kv::track::create();
          track->set_id( d->m_track_pairer.allocate_track_id() );
          track->append( state );
          output_trks2.push_back( track );
        }
      }
    }

    // Create output sets
    auto output_track_set1 = std::make_shared< kv::object_track_set >( output_trks1 );
    auto output_track_set2 = std::make_shared< kv::object_track_set >( output_trks2 );

    push_to_port_using_trait( object_track_set1, output_track_set1 );
    push_to_port_using_trait( object_track_set2, output_track_set2 );
  }
}


} // end namespace core
} // end namespace viame
