/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Detect shot breaks and create tracks for each shot
 */

#include "detect_shot_breaks_process.h"

#include <vital/algo/algorithm.txx>

#include "detect_shot_breaks.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/feature.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/match_set.h>

#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/match_features.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <cmath>
#include <algorithm>
#include <vector>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( fixed_frame_count, unsigned, "0",
  "If set, generate a full frame track for this many frames. "
  "Set to 0 to disable fixed-length tracks." );

create_config_trait( use_shot_break_detection, bool, "false",
  "If true, automatically detect shot breaks and start new tracks when "
  "significant visual changes occur. Requires the image input port to be connected." );

create_config_trait( shot_break_threshold, double, "0.3",
  "Threshold for shot break detection (0.0 to 1.0). Lower values are more "
  "sensitive to changes. A value of 0.3 means a 30% change in image content "
  "will trigger a new track." );

create_config_trait( shot_break_method, std::string, "histogram",
  "Method for detecting shot breaks. Options: "
  "'histogram' - Compare color histograms (good for lighting/color changes), "
  "'pixel_diff' - Compare mean absolute pixel difference (good for content changes), "
  "'combined' - Use histogram and pixel_diff methods together (most robust), "
  "'feature' - Use feature point matching (best for camera motion/content tracking), "
  "'descriptor' - Use global frame descriptor comparison (good for semantic changes)." );

create_config_trait( min_track_length, unsigned, "1",
  "Minimum number of frames before a shot break can trigger a new track. "
  "Prevents very short tracks from being created. Applies to all shot break methods." );

create_config_trait( max_track_length, unsigned, "0",
  "Maximum number of frames in a track before forcing a shot break. "
  "Set to 0 to disable maximum length enforcement. Applies to all shot break methods." );

create_config_trait( histogram_bins, unsigned, "32",
  "Number of bins per channel for histogram comparison. "
  "More bins = more sensitive to small color changes." );

create_config_trait( min_feature_matches, unsigned, "20",
  "Minimum number of feature matches between consecutive frames to consider "
  "them part of the same shot. If fewer matches are found, a shot break is triggered." );

create_config_trait( feature_match_ratio, double, "0.3",
  "Ratio of feature matches to total features detected in previous frame. "
  "If the ratio falls below this threshold, a shot break is triggered. "
  "Only used if min_feature_matches is also not met." );

create_config_trait( descriptor_distance_threshold, double, "0.5",
  "Maximum normalized distance between frame descriptors before triggering a shot break. "
  "Lower values are more sensitive to changes. The distance is normalized to 0-1 range "
  "based on the descriptor type." );

// =============================================================================
// Private implementation class
class detect_shot_breaks_process::priv
{
public:
  explicit priv( detect_shot_breaks_process* parent );
  ~priv();

  // Shot break detection methods
  bool detect_shot_break( const kv::image_container_sptr& current_image );
  bool detect_shot_break_features( const kv::image_container_sptr& current_image );
  bool detect_shot_break_descriptor( const kv::image_container_sptr& current_image );

  // Configuration settings
  unsigned m_fixed_frame_count;
  bool m_use_shot_break_detection;
  double m_shot_break_threshold;
  std::string m_shot_break_method;
  unsigned m_min_track_length;
  unsigned m_max_track_length;
  unsigned m_histogram_bins;
  unsigned m_min_feature_matches;
  double m_feature_match_ratio;
  double m_descriptor_distance_threshold;

  // Feature matching algorithms
  kv::algo::detect_features_sptr m_feature_detector;
  kv::algo::extract_descriptors_sptr m_descriptor_extractor;
  kv::algo::match_features_sptr m_feature_matcher;

  // Frame descriptor algorithm (for descriptor-based shot break)
  kv::algo::extract_descriptors_sptr m_frame_descriptor_extractor;

  // Internal variables
  unsigned m_frame_counter;
  unsigned m_track_counter;
  std::vector< kv::track_state_sptr > m_states;

  // Shot break detection state
  kv::image_container_sptr m_previous_image;
  std::vector< double > m_previous_histogram;

  // Feature-based shot break detection state
  kv::feature_set_sptr m_previous_features;
  kv::descriptor_set_sptr m_previous_descriptors;

  // Descriptor-based shot break detection state
  kv::descriptor_sptr m_previous_frame_descriptor;

  // Other variables
  detect_shot_breaks_process* parent;
};


// -----------------------------------------------------------------------------
detect_shot_breaks_process::priv
::priv( detect_shot_breaks_process* ptr )
  : m_fixed_frame_count( 0 )
  , m_use_shot_break_detection( false )
  , m_shot_break_threshold( 0.3 )
  , m_shot_break_method( "histogram" )
  , m_min_track_length( 1 )
  , m_max_track_length( 0 )
  , m_histogram_bins( 32 )
  , m_min_feature_matches( 20 )
  , m_feature_match_ratio( 0.3 )
  , m_descriptor_distance_threshold( 0.5 )
  , m_frame_counter( 0 )
  , m_track_counter( 1 )
  , parent( ptr )
{
}

// -----------------------------------------------------------------------------
detect_shot_breaks_process::priv
::~priv()
{
}

// -----------------------------------------------------------------------------
bool
detect_shot_breaks_process::priv
::detect_shot_break( const kv::image_container_sptr& current_image )
{
  if( !current_image || !m_use_shot_break_detection )
  {
    return false;
  }

  // Check maximum track length first - force shot break if exceeded
  if( m_max_track_length > 0 && m_states.size() >= m_max_track_length )
  {
    LOG_DEBUG( parent->logger(), "Maximum track length reached ("
               << m_max_track_length << " frames), forcing shot break" );
    return true;
  }

  // Check minimum track length - no shot break allowed yet, but update caches
  if( m_states.size() < m_min_track_length )
  {
    // Update caches for the appropriate method
    if( m_shot_break_method == "feature" )
    {
      if( m_feature_detector && m_descriptor_extractor )
      {
        m_previous_features = m_feature_detector->detect( current_image );
        if( m_previous_features )
        {
          m_previous_descriptors = m_descriptor_extractor->extract(
            current_image, m_previous_features );
        }
      }
    }
    else if( m_shot_break_method == "descriptor" )
    {
      if( m_frame_descriptor_extractor )
      {
        double cx = current_image->width() / 2.0;
        double cy = current_image->height() / 2.0;
        double scale = std::min( current_image->width(), current_image->height() ) / 2.0;
        auto center_feature = std::make_shared< kv::feature_d >(
          kv::vector_2d( cx, cy ), scale, 0.0 );
        std::vector< kv::feature_sptr > feature_vec;
        feature_vec.push_back( center_feature );
        kv::feature_set_sptr feature_set =
          std::make_shared< kv::simple_feature_set >( feature_vec );
        auto desc_set = m_frame_descriptor_extractor->extract( current_image, feature_set );
        if( desc_set && desc_set->size() > 0 )
        {
          m_previous_frame_descriptor = desc_set->descriptors()[0];
        }
      }
    }
    else
    {
      // histogram, pixel_diff, or combined methods
      m_previous_image = current_image;
      m_previous_histogram = compute_image_histogram( current_image, m_histogram_bins );
    }
    return false;
  }

  // Dispatch to method-specific detection
  if( m_shot_break_method == "feature" )
  {
    return detect_shot_break_features( current_image );
  }

  if( m_shot_break_method == "descriptor" )
  {
    return detect_shot_break_descriptor( current_image );
  }

  // If no previous image, this is the first frame - no shot break
  if( !m_previous_image )
  {
    m_previous_image = current_image;
    m_previous_histogram = compute_image_histogram( current_image, m_histogram_bins );
    return false;
  }

  double change_score = 0.0;

  if( m_shot_break_method == "histogram" )
  {
    // Use cached histogram if available for efficiency
    std::vector< double > current_hist = compute_image_histogram( current_image, m_histogram_bins );

    if( !m_previous_histogram.empty() && !current_hist.empty() &&
        m_previous_histogram.size() == current_hist.size() )
    {
      double intersection = 0.0;
      for( size_t i = 0; i < current_hist.size(); ++i )
      {
        intersection += std::min( m_previous_histogram[i], current_hist[i] );
      }
      change_score = 1.0 - intersection;
    }

    // Update cached histogram
    m_previous_histogram = current_hist;
  }
  else if( m_shot_break_method == "pixel_diff" )
  {
    change_score = compute_pixel_difference( m_previous_image, current_image );
  }
  else // combined
  {
    double hist_diff = compute_histogram_difference( m_previous_image, current_image, m_histogram_bins );
    double pixel_diff = compute_pixel_difference( m_previous_image, current_image );

    // Use maximum of both methods
    change_score = std::max( hist_diff, pixel_diff );

    // Update cached histogram for next frame
    m_previous_histogram = compute_image_histogram( current_image, m_histogram_bins );
  }

  // Update previous image reference
  m_previous_image = current_image;

  // Check threshold
  bool is_shot_break = ( change_score >= m_shot_break_threshold );

  if( is_shot_break )
  {
    LOG_DEBUG( parent->logger(), "Shot break detected with score "
               << change_score << " >= threshold " << m_shot_break_threshold
               << " (track had " << m_states.size() << " frames)" );
  }

  return is_shot_break;
}

// -----------------------------------------------------------------------------
bool
detect_shot_breaks_process::priv
::detect_shot_break_features( const kv::image_container_sptr& current_image )
{
  if( !current_image )
  {
    return false;
  }

  // Check if feature matching algorithms are configured
  if( !m_feature_detector || !m_descriptor_extractor || !m_feature_matcher )
  {
    LOG_WARN( parent->logger(), "Feature-based shot break detection requires "
              "feature_detector, descriptor_extractor, and feature_matcher algorithms. "
              "Falling back to histogram method." );
    return false;
  }

  // Detect features in current image
  kv::feature_set_sptr current_features = m_feature_detector->detect( current_image );

  if( !current_features || current_features->size() == 0 )
  {
    // No features detected - could be a blank frame or scene change
    m_previous_features = nullptr;
    m_previous_descriptors = nullptr;
    return true;
  }

  // Extract descriptors for current features
  kv::descriptor_set_sptr current_descriptors =
    m_descriptor_extractor->extract( current_image, current_features );

  if( !current_descriptors || current_descriptors->size() == 0 )
  {
    m_previous_features = current_features;
    m_previous_descriptors = nullptr;
    return true;
  }

  // If no previous frame data, this is the first frame
  if( !m_previous_features || !m_previous_descriptors )
  {
    m_previous_features = current_features;
    m_previous_descriptors = current_descriptors;
    return false;
  }

  // Match features between previous and current frame
  kv::match_set_sptr matches = m_feature_matcher->match(
    m_previous_features, m_previous_descriptors,
    current_features, current_descriptors );

  size_t num_matches = matches ? matches->size() : 0;
  size_t prev_feature_count = m_previous_features->size();

  // Update cached features for next frame
  m_previous_features = current_features;
  m_previous_descriptors = current_descriptors;

  // Check shot break criteria
  bool is_shot_break = false;

  // Primary criterion: minimum number of matches
  if( num_matches < m_min_feature_matches )
  {
    is_shot_break = true;
  }

  // Secondary criterion: ratio of matches to previous features
  if( prev_feature_count > 0 )
  {
    double match_ratio = static_cast< double >( num_matches ) / prev_feature_count;
    if( match_ratio < m_feature_match_ratio )
    {
      is_shot_break = true;
    }
  }

  if( is_shot_break )
  {
    LOG_DEBUG( parent->logger(), "Feature-based shot break detected: "
               << num_matches << " matches (min: " << m_min_feature_matches
               << "), ratio: " << ( prev_feature_count > 0 ?
                  static_cast< double >( num_matches ) / prev_feature_count : 0.0 )
               << " (min: " << m_feature_match_ratio << ")"
               << " (track had " << m_states.size() << " frames)" );
  }

  return is_shot_break;
}

// -----------------------------------------------------------------------------
bool
detect_shot_breaks_process::priv
::detect_shot_break_descriptor( const kv::image_container_sptr& current_image )
{
  if( !current_image )
  {
    return false;
  }

  // Check if frame descriptor algorithm is configured
  if( !m_frame_descriptor_extractor )
  {
    LOG_WARN( parent->logger(), "Descriptor-based shot break detection requires "
              "'frame_descriptor_extractor' algorithm. Falling back to histogram method." );
    return false;
  }

  // Create a single feature at the center of the image to extract a global descriptor
  double cx = current_image->width() / 2.0;
  double cy = current_image->height() / 2.0;
  double scale = std::min( current_image->width(), current_image->height() ) / 2.0;

  auto center_feature = std::make_shared< kv::feature_d >(
    kv::vector_2d( cx, cy ), scale, 0.0 );

  std::vector< kv::feature_sptr > feature_vec;
  feature_vec.push_back( center_feature );
  kv::feature_set_sptr feature_set =
    std::make_shared< kv::simple_feature_set >( feature_vec );

  // Extract descriptor for the center feature (should give global descriptor)
  kv::descriptor_set_sptr desc_set =
    m_frame_descriptor_extractor->extract( current_image, feature_set );

  if( !desc_set || desc_set->size() == 0 )
  {
    LOG_WARN( parent->logger(), "Failed to extract frame descriptor" );
    m_previous_frame_descriptor = nullptr;
    return true; // Treat as shot break if we can't get a descriptor
  }

  kv::descriptor_sptr current_descriptor = desc_set->descriptors()[0];

  // If no previous descriptor, this is the first frame
  if( !m_previous_frame_descriptor )
  {
    m_previous_frame_descriptor = current_descriptor;
    return false;
  }

  // Compute distance between descriptors
  double distance = compute_descriptor_distance( m_previous_frame_descriptor, current_descriptor );

  // Update cached descriptor
  m_previous_frame_descriptor = current_descriptor;

  // Check threshold
  bool is_shot_break = ( distance >= m_descriptor_distance_threshold );

  if( is_shot_break )
  {
    LOG_DEBUG( parent->logger(), "Descriptor-based shot break detected: distance "
               << distance << " >= threshold " << m_descriptor_distance_threshold
               << " (track had " << m_states.size() << " frames)" );
  }

  return is_shot_break;
}


// =============================================================================
detect_shot_breaks_process
::detect_shot_breaks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new detect_shot_breaks_process::priv( this ) )
{
  make_ports();
  make_config();
}


detect_shot_breaks_process
::~detect_shot_breaks_process()
{
}


// -----------------------------------------------------------------------------
void
detect_shot_breaks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( detected_object_set, required );

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( object_track_set, optional );
}

// -----------------------------------------------------------------------------
void
detect_shot_breaks_process
::make_config()
{
  declare_config_using_trait( fixed_frame_count );
  declare_config_using_trait( use_shot_break_detection );
  declare_config_using_trait( shot_break_threshold );
  declare_config_using_trait( shot_break_method );
  declare_config_using_trait( min_track_length );
  declare_config_using_trait( max_track_length );
  declare_config_using_trait( histogram_bins );
  declare_config_using_trait( min_feature_matches );
  declare_config_using_trait( feature_match_ratio );
  declare_config_using_trait( descriptor_distance_threshold );
}

// -----------------------------------------------------------------------------
void
detect_shot_breaks_process
::_configure()
{
  d->m_fixed_frame_count = config_value_using_trait( fixed_frame_count );
  d->m_use_shot_break_detection = config_value_using_trait( use_shot_break_detection );
  d->m_shot_break_threshold = config_value_using_trait( shot_break_threshold );
  d->m_shot_break_method = config_value_using_trait( shot_break_method );
  d->m_min_track_length = config_value_using_trait( min_track_length );
  d->m_max_track_length = config_value_using_trait( max_track_length );
  d->m_histogram_bins = config_value_using_trait( histogram_bins );
  d->m_min_feature_matches = config_value_using_trait( min_feature_matches );
  d->m_feature_match_ratio = config_value_using_trait( feature_match_ratio );
  d->m_descriptor_distance_threshold = config_value_using_trait( descriptor_distance_threshold );

  // Validate configuration
  if( d->m_shot_break_threshold < 0.0 || d->m_shot_break_threshold > 1.0 )
  {
    LOG_WARN( logger(), "shot_break_threshold should be between 0.0 and 1.0, "
              "clamping value " << d->m_shot_break_threshold );
    d->m_shot_break_threshold = std::max( 0.0, std::min( 1.0, d->m_shot_break_threshold ) );
  }

  if( d->m_shot_break_method != "histogram" &&
      d->m_shot_break_method != "pixel_diff" &&
      d->m_shot_break_method != "combined" &&
      d->m_shot_break_method != "feature" &&
      d->m_shot_break_method != "descriptor" )
  {
    LOG_WARN( logger(), "Invalid shot_break_method '" << d->m_shot_break_method
              << "', defaulting to 'histogram'" );
    d->m_shot_break_method = "histogram";
  }

  // Validate min/max track length consistency
  if( d->m_max_track_length > 0 && d->m_max_track_length < d->m_min_track_length )
  {
    LOG_WARN( logger(), "max_track_length (" << d->m_max_track_length
              << ") is less than min_track_length (" << d->m_min_track_length
              << "), setting max_track_length to min_track_length" );
    d->m_max_track_length = d->m_min_track_length;
  }

  if( d->m_histogram_bins < 4 || d->m_histogram_bins > 256 )
  {
    LOG_WARN( logger(), "histogram_bins should be between 4 and 256, "
              "clamping value " << d->m_histogram_bins );
    d->m_histogram_bins = std::max( 4u, std::min( 256u, d->m_histogram_bins ) );
  }

  if( d->m_feature_match_ratio < 0.0 || d->m_feature_match_ratio > 1.0 )
  {
    LOG_WARN( logger(), "feature_match_ratio should be between 0.0 and 1.0, "
              "clamping value " << d->m_feature_match_ratio );
    d->m_feature_match_ratio = std::max( 0.0, std::min( 1.0, d->m_feature_match_ratio ) );
  }

  if( d->m_descriptor_distance_threshold < 0.0 || d->m_descriptor_distance_threshold > 1.0 )
  {
    LOG_WARN( logger(), "descriptor_distance_threshold should be between 0.0 and 1.0, "
              "clamping value " << d->m_descriptor_distance_threshold );
    d->m_descriptor_distance_threshold = std::max( 0.0, std::min( 1.0, d->m_descriptor_distance_threshold ) );
  }

  // Set up feature matching algorithms if using feature-based method
  if( d->m_use_shot_break_detection && d->m_shot_break_method == "feature" )
  {
    kv::config_block_sptr algo_config = get_config();

    kv::set_nested_algo_configuration<kv::algo::detect_features>(
      "feature_detector", algo_config, d->m_feature_detector );
    kv::set_nested_algo_configuration<kv::algo::extract_descriptors>(
      "descriptor_extractor", algo_config, d->m_descriptor_extractor );
    kv::set_nested_algo_configuration<kv::algo::match_features>(
      "feature_matcher", algo_config, d->m_feature_matcher );

    if( !d->m_feature_detector )
    {
      LOG_ERROR( logger(), "Feature-based shot break detection requires "
                 "'feature_detector' algorithm to be configured." );
    }
    if( !d->m_descriptor_extractor )
    {
      LOG_ERROR( logger(), "Feature-based shot break detection requires "
                 "'descriptor_extractor' algorithm to be configured." );
    }
    if( !d->m_feature_matcher )
    {
      LOG_ERROR( logger(), "Feature-based shot break detection requires "
                 "'feature_matcher' algorithm to be configured." );
    }
  }

  // Set up frame descriptor algorithm if using descriptor-based method
  if( d->m_use_shot_break_detection && d->m_shot_break_method == "descriptor" )
  {
    kv::config_block_sptr algo_config = get_config();

    kv::set_nested_algo_configuration<kv::algo::extract_descriptors>(
      "frame_descriptor_extractor", algo_config, d->m_frame_descriptor_extractor );

    if( !d->m_frame_descriptor_extractor )
    {
      LOG_ERROR( logger(), "Descriptor-based shot break detection requires "
                 "'frame_descriptor_extractor' algorithm to be configured." );
    }
  }

  // Log configuration
  if( d->m_use_shot_break_detection )
  {
    LOG_INFO( logger(), "Shot break detection enabled:" );
    LOG_INFO( logger(), "  Method: " << d->m_shot_break_method );
    LOG_INFO( logger(), "  Min track length: " << d->m_min_track_length );
    if( d->m_max_track_length > 0 )
    {
      LOG_INFO( logger(), "  Max track length: " << d->m_max_track_length );
    }
    else
    {
      LOG_INFO( logger(), "  Max track length: unlimited" );
    }

    if( d->m_shot_break_method == "histogram" ||
        d->m_shot_break_method == "pixel_diff" ||
        d->m_shot_break_method == "combined" )
    {
      LOG_INFO( logger(), "  Threshold: " << d->m_shot_break_threshold );
    }
    if( d->m_shot_break_method == "histogram" || d->m_shot_break_method == "combined" )
    {
      LOG_INFO( logger(), "  Histogram bins: " << d->m_histogram_bins );
    }
    if( d->m_shot_break_method == "feature" )
    {
      LOG_INFO( logger(), "  Min feature matches: " << d->m_min_feature_matches );
      LOG_INFO( logger(), "  Feature match ratio: " << d->m_feature_match_ratio );
    }
    if( d->m_shot_break_method == "descriptor" )
    {
      LOG_INFO( logger(), "  Descriptor distance threshold: " << d->m_descriptor_distance_threshold );
    }
  }
}

// -----------------------------------------------------------------------------
void
detect_shot_breaks_process
::_step()
{
  kv::image_container_sptr image;
  kv::timestamp timestamp;
  kv::detected_object_set_sptr detections;

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    timestamp = grab_from_port_using_trait( timestamp );
  }
  if( has_input_port_edge_using_trait( image ) )
  {
    image = grab_from_port_using_trait( image );
  }
  if( has_input_port_edge_using_trait( detected_object_set ) )
  {
    detections = grab_from_port_using_trait( detected_object_set );
  }

  // Check for shot break (requires image input)
  bool shot_break = false;
  if( d->m_use_shot_break_detection && image )
  {
    shot_break = d->detect_shot_break( image );
  }
  else if( d->m_use_shot_break_detection && !image )
  {
    // Warn only once about missing image input
    static bool warned = false;
    if( !warned )
    {
      LOG_WARN( logger(), "Shot break detection enabled but image input is not connected. "
                "Shot break detection will be disabled." );
      warned = true;
    }
  }

  // Start new track if:
  // 1. Fixed frame count reached (if configured)
  // 2. Shot break detected
  bool start_new_track = false;

  if( d->m_fixed_frame_count > 0 && d->m_states.size() >= d->m_fixed_frame_count )
  {
    start_new_track = true;
  }

  if( shot_break )
  {
    start_new_track = true;
  }

  if( start_new_track && !d->m_states.empty() )
  {
    d->m_track_counter++;
    d->m_states.clear();

    // Reset shot break detection state for new track
    if( shot_break )
    {
      d->m_previous_histogram.clear();
      // For feature method, clear cached features so next frame starts fresh
      // (detect_shot_break_features will repopulate with current frame)
      d->m_previous_features = nullptr;
      d->m_previous_descriptors = nullptr;
      // For descriptor method, clear cached frame descriptor
      d->m_previous_frame_descriptor = nullptr;
      // Keep the current image as reference for the new track
    }
  }

  if( detections->size() == 1 )
  {
    d->m_states.push_back(
      std::make_shared< kwiver::vital::object_track_state >(
        timestamp, detections->at( 0 ) ) );
  }

  kv::track_sptr ot = kv::track::create();
  ot->set_id( d->m_track_counter );

  for( auto state : d->m_states )
  {
    ot->append( state );
  }

  kv::object_track_set_sptr output(
    new kv::object_track_set(
      std::vector< kv::track_sptr >( 1, ot ) ) );

  push_to_port_using_trait( timestamp, timestamp );
  push_to_port_using_trait( object_track_set, output );
}

} // end namespace core

} // end namespace viame
