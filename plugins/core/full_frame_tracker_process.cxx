/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Consolidate the output of multiple object trackers
 */

#include "full_frame_tracker_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
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
  "'feature' - Use feature point matching (best for camera motion/content tracking)." );

create_config_trait( min_track_length, unsigned, "1",
  "Minimum number of frames before a shot break can trigger a new track. "
  "Prevents very short tracks from being created." );

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

// =============================================================================
// Private implementation class
class full_frame_tracker_process::priv
{
public:
  explicit priv( full_frame_tracker_process* parent );
  ~priv();

  // Shot break detection methods
  bool detect_shot_break( const kv::image_container_sptr& current_image );
  double compute_histogram_difference( const kv::image_container_sptr& img1,
                                        const kv::image_container_sptr& img2 ) const;
  double compute_pixel_difference( const kv::image_container_sptr& img1,
                                    const kv::image_container_sptr& img2 ) const;
  std::vector< double > compute_histogram( const kv::image_container_sptr& img ) const;
  bool detect_shot_break_features( const kv::image_container_sptr& current_image );

  // Configuration settings
  unsigned m_fixed_frame_count;
  bool m_use_shot_break_detection;
  double m_shot_break_threshold;
  std::string m_shot_break_method;
  unsigned m_min_track_length;
  unsigned m_histogram_bins;
  unsigned m_min_feature_matches;
  double m_feature_match_ratio;

  // Feature matching algorithms
  kv::algo::detect_features_sptr m_feature_detector;
  kv::algo::extract_descriptors_sptr m_descriptor_extractor;
  kv::algo::match_features_sptr m_feature_matcher;

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

  // Other variables
  full_frame_tracker_process* parent;
};


// -----------------------------------------------------------------------------
full_frame_tracker_process::priv
::priv( full_frame_tracker_process* ptr )
  : m_fixed_frame_count( 0 )
  , m_use_shot_break_detection( false )
  , m_shot_break_threshold( 0.3 )
  , m_shot_break_method( "histogram" )
  , m_min_track_length( 1 )
  , m_histogram_bins( 32 )
  , m_min_feature_matches( 20 )
  , m_feature_match_ratio( 0.3 )
  , m_frame_counter( 0 )
  , m_track_counter( 1 )
  , parent( ptr )
{
}

// -----------------------------------------------------------------------------
full_frame_tracker_process::priv
::~priv()
{
}

// -----------------------------------------------------------------------------
std::vector< double >
full_frame_tracker_process::priv
::compute_histogram( const kv::image_container_sptr& img ) const
{
  if( !img )
  {
    return std::vector< double >();
  }

  const kv::image& image = img->get_image();
  size_t width = image.width();
  size_t height = image.height();
  size_t depth = image.depth();

  // Create histogram bins for each channel
  size_t total_bins = m_histogram_bins * depth;
  std::vector< double > histogram( total_bins, 0.0 );

  // Sample pixels (use stride for large images to improve performance)
  size_t stride = std::max( size_t( 1 ), std::min( width, height ) / 100 );
  size_t sample_count = 0;

  for( size_t y = 0; y < height; y += stride )
  {
    for( size_t x = 0; x < width; x += stride )
    {
      for( size_t c = 0; c < depth; ++c )
      {
        uint8_t pixel_val = image.at< uint8_t >( x, y, c );
        size_t bin = static_cast< size_t >( pixel_val ) * m_histogram_bins / 256;
        bin = std::min( bin, static_cast< size_t >( m_histogram_bins - 1 ) );
        histogram[c * m_histogram_bins + bin] += 1.0;
      }
      sample_count++;
    }
  }

  // Normalize histogram
  if( sample_count > 0 )
  {
    for( auto& val : histogram )
    {
      val /= static_cast< double >( sample_count );
    }
  }

  return histogram;
}

// -----------------------------------------------------------------------------
double
full_frame_tracker_process::priv
::compute_histogram_difference( const kv::image_container_sptr& img1,
                                 const kv::image_container_sptr& img2 ) const
{
  auto hist1 = compute_histogram( img1 );
  auto hist2 = compute_histogram( img2 );

  if( hist1.empty() || hist2.empty() || hist1.size() != hist2.size() )
  {
    return 0.0;
  }

  // Compute histogram intersection (similarity measure)
  // Intersection = sum of min values, result is 0-1 where 1 = identical
  double intersection = 0.0;
  for( size_t i = 0; i < hist1.size(); ++i )
  {
    intersection += std::min( hist1[i], hist2[i] );
  }

  // Convert to difference (0 = identical, 1 = completely different)
  return 1.0 - intersection;
}

// -----------------------------------------------------------------------------
double
full_frame_tracker_process::priv
::compute_pixel_difference( const kv::image_container_sptr& img1,
                             const kv::image_container_sptr& img2 ) const
{
  if( !img1 || !img2 )
  {
    return 0.0;
  }

  const kv::image& image1 = img1->get_image();
  const kv::image& image2 = img2->get_image();

  // Check dimensions match
  if( image1.width() != image2.width() ||
      image1.height() != image2.height() ||
      image1.depth() != image2.depth() )
  {
    // Different dimensions = scene change
    return 1.0;
  }

  size_t width = image1.width();
  size_t height = image1.height();
  size_t depth = image1.depth();

  // Sample pixels (use stride for large images to improve performance)
  size_t stride = std::max( size_t( 1 ), std::min( width, height ) / 100 );
  double total_diff = 0.0;
  size_t sample_count = 0;

  for( size_t y = 0; y < height; y += stride )
  {
    for( size_t x = 0; x < width; x += stride )
    {
      for( size_t c = 0; c < depth; ++c )
      {
        int val1 = static_cast< int >( image1.at< uint8_t >( x, y, c ) );
        int val2 = static_cast< int >( image2.at< uint8_t >( x, y, c ) );
        total_diff += std::abs( val1 - val2 );
      }
      sample_count++;
    }
  }

  // Normalize to 0-1 range (max difference is 255 per channel)
  if( sample_count > 0 && depth > 0 )
  {
    return total_diff / ( sample_count * depth * 255.0 );
  }

  return 0.0;
}

// -----------------------------------------------------------------------------
bool
full_frame_tracker_process::priv
::detect_shot_break( const kv::image_container_sptr& current_image )
{
  if( !current_image || !m_use_shot_break_detection )
  {
    return false;
  }

  // Feature-based detection has its own min track length check
  if( m_shot_break_method == "feature" )
  {
    // Check minimum track length
    if( m_states.size() < m_min_track_length )
    {
      // Still need to update feature cache for first frames
      if( m_feature_detector && m_descriptor_extractor )
      {
        m_previous_features = m_feature_detector->detect( current_image );
        if( m_previous_features )
        {
          m_previous_descriptors = m_descriptor_extractor->extract(
            current_image, m_previous_features );
        }
      }
      return false;
    }
    return detect_shot_break_features( current_image );
  }

  // Check minimum track length for other methods
  if( m_states.size() < m_min_track_length )
  {
    // Update previous image for next comparison
    m_previous_image = current_image;
    m_previous_histogram = compute_histogram( current_image );
    return false;
  }

  // If no previous image, this is the first frame - no shot break
  if( !m_previous_image )
  {
    m_previous_image = current_image;
    m_previous_histogram = compute_histogram( current_image );
    return false;
  }

  double change_score = 0.0;

  if( m_shot_break_method == "histogram" )
  {
    // Use cached histogram if available for efficiency
    std::vector< double > current_hist = compute_histogram( current_image );

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
    double hist_diff = compute_histogram_difference( m_previous_image, current_image );
    double pixel_diff = compute_pixel_difference( m_previous_image, current_image );

    // Use maximum of both methods
    change_score = std::max( hist_diff, pixel_diff );

    // Update cached histogram for next frame
    m_previous_histogram = compute_histogram( current_image );
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
full_frame_tracker_process::priv
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


// =============================================================================
full_frame_tracker_process
::full_frame_tracker_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new full_frame_tracker_process::priv( this ) )
{
  make_ports();
  make_config();
}


full_frame_tracker_process
::~full_frame_tracker_process()
{
}


// -----------------------------------------------------------------------------
void
full_frame_tracker_process
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
full_frame_tracker_process
::make_config()
{
  declare_config_using_trait( fixed_frame_count );
  declare_config_using_trait( use_shot_break_detection );
  declare_config_using_trait( shot_break_threshold );
  declare_config_using_trait( shot_break_method );
  declare_config_using_trait( min_track_length );
  declare_config_using_trait( histogram_bins );
  declare_config_using_trait( min_feature_matches );
  declare_config_using_trait( feature_match_ratio );
}

// -----------------------------------------------------------------------------
void
full_frame_tracker_process
::_configure()
{
  d->m_fixed_frame_count = config_value_using_trait( fixed_frame_count );
  d->m_use_shot_break_detection = config_value_using_trait( use_shot_break_detection );
  d->m_shot_break_threshold = config_value_using_trait( shot_break_threshold );
  d->m_shot_break_method = config_value_using_trait( shot_break_method );
  d->m_min_track_length = config_value_using_trait( min_track_length );
  d->m_histogram_bins = config_value_using_trait( histogram_bins );
  d->m_min_feature_matches = config_value_using_trait( min_feature_matches );
  d->m_feature_match_ratio = config_value_using_trait( feature_match_ratio );

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
      d->m_shot_break_method != "feature" )
  {
    LOG_WARN( logger(), "Invalid shot_break_method '" << d->m_shot_break_method
              << "', defaulting to 'histogram'" );
    d->m_shot_break_method = "histogram";
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

  // Set up feature matching algorithms if using feature-based method
  if( d->m_use_shot_break_detection && d->m_shot_break_method == "feature" )
  {
    kv::config_block_sptr algo_config = get_config();

    kv::algo::detect_features::set_nested_algo_configuration(
      "feature_detector", algo_config, d->m_feature_detector );
    kv::algo::extract_descriptors::set_nested_algo_configuration(
      "descriptor_extractor", algo_config, d->m_descriptor_extractor );
    kv::algo::match_features::set_nested_algo_configuration(
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

  // Log configuration
  if( d->m_use_shot_break_detection )
  {
    LOG_INFO( logger(), "Shot break detection enabled:" );
    LOG_INFO( logger(), "  Method: " << d->m_shot_break_method );
    LOG_INFO( logger(), "  Min track length: " << d->m_min_track_length );

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
  }
}

// -----------------------------------------------------------------------------
void
full_frame_tracker_process
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
