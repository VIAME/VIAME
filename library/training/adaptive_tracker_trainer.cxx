/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "adaptive_tracker_trainer.h"

#include <viame/algorithm_framework/algo/algorithm.txx>
#include <viame/algorithm_framework/algo/feature_descriptor_io.h>
#include <viame/algorithm_framework/util/cpu_timer.h>
#include <viame/algorithm_framework/exceptions.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/object_track_set.h>

#include <viame/core_types/math/matrix.h>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>
#include <set>

namespace viame {

namespace kv = viame;

// =============================================================================
// Statistics structure for tracking data analysis
struct tracking_data_statistics
{
  // -------------------------------------------------------------------------
  // Track counts
  size_t total_tracks = 0;
  size_t total_train_tracks = 0;
  size_t total_test_tracks = 0;
  size_t total_detections = 0;  // Sum of all track states

  // Per-class track counts
  std::map< std::string, size_t > class_track_counts;
  std::map< std::string, size_t > class_detection_counts;

  // -------------------------------------------------------------------------
  // Track length statistics
  std::vector< size_t > track_lengths;
  double min_track_length = 0;
  double max_track_length = 0;
  double mean_track_length = 0;
  double median_track_length = 0;

  // Track length distribution
  size_t short_track_count = 0;    // < short_threshold frames
  size_t medium_track_count = 0;   // >= short, < long threshold
  size_t long_track_count = 0;     // >= long_threshold frames

  // -------------------------------------------------------------------------
  // Track fragmentation (gaps within tracks)
  std::vector< size_t > gaps_per_track;
  std::vector< size_t > gap_lengths;
  double mean_gaps_per_track = 0;
  double mean_gap_length = 0;
  size_t tracks_with_gaps = 0;
  double fragmentation_rate = 0;  // fraction of tracks with gaps

  // -------------------------------------------------------------------------
  // Motion statistics
  std::vector< double > velocities;  // pixels per frame
  std::vector< double > direction_changes;  // angle changes in radians
  double mean_velocity = 0;
  double max_velocity = 0;
  double velocity_std = 0;
  double mean_direction_change = 0;

  // Motion classification
  size_t stationary_track_count = 0;  // very low velocity
  size_t slow_track_count = 0;
  size_t fast_track_count = 0;

  // -------------------------------------------------------------------------
  // Concurrent tracks (temporal density)
  std::vector< size_t > concurrent_tracks_per_frame;
  double mean_concurrent_tracks = 0;
  double max_concurrent_tracks = 0;
  size_t crowded_frame_count = 0;
  size_t sparse_frame_count = 0;

  // -------------------------------------------------------------------------
  // Object sizes (for Re-ID decisions)
  std::vector< double > object_areas;
  std::vector< double > object_widths;
  std::vector< double > object_heights;
  double mean_object_area = 0;
  double min_object_area = 0;
  double max_object_area = 0;

  // Size distribution
  size_t small_object_count = 0;
  size_t medium_object_count = 0;
  size_t large_object_count = 0;

  // -------------------------------------------------------------------------
  // Appearance consistency (size variance within tracks)
  std::vector< double > within_track_size_variance;
  double mean_size_variance = 0;
  double high_variance_track_fraction = 0;

  // -------------------------------------------------------------------------
  // Track proximity/occlusion analysis
  double mean_min_track_distance = 0;  // Mean of min distances between concurrent tracks
  size_t close_track_pair_count = 0;   // Pairs of tracks that get very close
  double occlusion_prone_fraction = 0; // Fraction of tracks with close encounters

  // -------------------------------------------------------------------------
  // ID switches potential (track crossings)
  size_t potential_id_switch_count = 0;
  double id_switch_rate = 0;  // per frame

  // -------------------------------------------------------------------------
  // Association: how well each motion model carries a box to its next state
  std::vector< double > raw_ious;         // previous box as-is
  std::vector< double > predicted_ious;   // constant-velocity prediction
  std::vector< double > registered_ious;  // previous box moved by the other tracks' fitted transform
  std::vector< double > registered_residuals;  // leftover motion after that, in box sizes
  size_t association_pair_count = 0;
  size_t ambiguous_pair_count = 0;
  double camera_motion_energy = 0;
  double total_motion_energy = 0;

  double median_raw_iou = 0;
  double median_predicted_iou = 0;
  double median_registered_iou = 0;
  double median_registered_residual = 0;
  double iou_match_rate = 0;
  double motion_match_rate = 0;
  double registration_match_rate = 0;
  double registration_coverage = 0;
  double camera_motion_fraction = 0;
  double ambiguity_rate = 0;

  // "iou", "motion", "registration" (objects move within the registered
  // frame), "fixed" (they don't), "appearance", or empty with no pairs
  std::string association_regime;

  // -------------------------------------------------------------------------
  // Frame statistics
  size_t total_frames = 0;
  size_t frames_with_tracks = 0;

  // -------------------------------------------------------------------------
  // Methods
  void compute_summary(
    size_t short_thresh, size_t long_thresh,
    double stationary_vel_thresh, double fast_vel_thresh,
    size_t crowded_thresh, size_t sparse_thresh,
    double small_area_thresh, double large_area_thresh,
    double close_distance_thresh, double high_variance_thresh );

  void classify_association(
    double match_iou, double required_match_rate,
    double max_ambiguity_rate, double min_registration_coverage,
    double registration_margin, double fixed_residual );

  void log_statistics( kv::logger_handle_t logger ) const;
};


namespace {

// Full pivoting for the three affine normal-equation coefficients. Keep the
// upstream relative pivot threshold, including its translation fallback for
// collinear points, without reintroducing Eigen into the lite build.
class affine_lu
{
public:
  explicit affine_lu( kv::matrix_3x3d matrix ) : m_lu( matrix )
  {
    for( unsigned k = 0; k < 3; ++k )
    {
      unsigned row = k, col = k;
      double largest = 0.0;
      for( unsigned j = k; j < 3; ++j )
      {
        for( unsigned i = k; i < 3; ++i )
        {
          if( std::abs( m_lu( i, j ) ) > largest )
          {
            largest = std::abs( m_lu( i, j ) ); row = i; col = j;
          }
        }
      }
      m_rows[k] = row;
      if( largest == 0.0 ) { break; }
      for( unsigned j = 0; j < 3; ++j )
      { std::swap( m_lu( k, j ), m_lu( row, j ) ); }
      for( unsigned i = 0; i < 3; ++i )
      { std::swap( m_lu( i, k ), m_lu( i, col ) ); }
      std::swap( m_columns[k], m_columns[col] );
      m_max_pivot = std::max( m_max_pivot, largest );
      for( unsigned i = k + 1; i < 3; ++i )
      {
        m_lu( i, k ) /= m_lu( k, k );
        for( unsigned j = k + 1; j < 3; ++j )
        { m_lu( i, j ) -= m_lu( i, k ) * m_lu( k, j ); }
      }
    }
  }

  void setThreshold( double threshold ) { m_threshold = threshold; }
  unsigned rank() const
  {
    unsigned result = 0;
    for( unsigned i = 0; i < 3; ++i )
    { result += std::abs( m_lu( i, i ) ) > m_threshold * m_max_pivot; }
    return result;
  }
  kv::vector_3d solve( kv::vector_3d rhs ) const
  {
    for( unsigned k = 0; k < 3; ++k )
    { std::swap( rhs[k], rhs[m_rows[k]] ); }
    for( unsigned i = 0; i < 3; ++i )
    { for( unsigned j = 0; j < i; ++j ) { rhs[i] -= m_lu( i, j ) * rhs[j]; } }
    for( int i = 2; i >= 0; --i )
    {
      for( unsigned j = i + 1; j < 3; ++j ) { rhs[i] -= m_lu( i, j ) * rhs[j]; }
      rhs[i] /= m_lu( i, i );
    }
    kv::vector_3d result;
    for( unsigned i = 0; i < 3; ++i ) { result[m_columns[i]] = rhs[i]; }
    return result;
  }
private:
  kv::matrix_3x3d m_lu;
  unsigned m_rows[3] = { 0, 1, 2 };
  unsigned m_columns[3] = { 0, 1, 2 };
  double m_max_pivot = 0.0;
  double m_threshold = 1e-9;
};


double
box_iou( const kv::bounding_box_d& a, const kv::bounding_box_d& b )
{
  double iw = std::min( a.max_x(), b.max_x() ) - std::max( a.min_x(), b.min_x() );
  double ih = std::min( a.max_y(), b.max_y() ) - std::max( a.min_y(), b.min_y() );

  if( iw <= 0 || ih <= 0 )
  {
    return 0.0;
  }

  double inter = iw * ih;
  double uni = a.area() + b.area() - inter;
  return uni > 0 ? inter / uni : 0.0;
}


kv::bounding_box_d
shifted_box( const kv::bounding_box_d& b, double dx, double dy )
{
  return kv::bounding_box_d(
    b.min_x() + dx, b.min_y() + dy, b.max_x() + dx, b.max_y() + dy );
}


double
median_of( std::vector< double > values )
{
  if( values.empty() )
  {
    return 0.0;
  }

  auto mid = values.begin() + values.size() / 2;
  std::nth_element( values.begin(), mid, values.end() );
  return *mid;
}


double
fraction_at_least( const std::vector< double >& values, double thresh )
{
  if( values.empty() )
  {
    return 0.0;
  }

  size_t count = std::count_if( values.begin(), values.end(),
    [thresh]( double v ){ return v >= thresh; } );
  return static_cast< double >( count ) / values.size();
}

} // end anonymous namespace


void
tracking_data_statistics::classify_association(
  double match_iou, double required_match_rate,
  double max_ambiguity_rate, double min_registration_coverage,
  double registration_margin, double fixed_residual )
{
  if( association_pair_count == 0 )
  {
    return;
  }

  median_raw_iou = median_of( raw_ious );
  median_predicted_iou = median_of( predicted_ious );
  median_registered_iou = median_of( registered_ious );
  median_registered_residual = median_of( registered_residuals );

  iou_match_rate = fraction_at_least( raw_ious, match_iou );
  motion_match_rate = fraction_at_least( predicted_ious, match_iou );
  registration_match_rate = fraction_at_least( registered_ious, match_iou );

  registration_coverage =
    static_cast< double >( registered_ious.size() ) / association_pair_count;
  ambiguity_rate =
    static_cast< double >( ambiguous_pair_count ) / association_pair_count;

  if( total_motion_energy > 0 )
  {
    camera_motion_fraction =
      std::max( 0.0, camera_motion_energy / total_motion_energy );
  }

  // Ambiguity is checked first: no motion model fixes two objects
  // overlapping the same predicted box.
  if( ambiguity_rate > max_ambiguity_rate )
    association_regime = "appearance";
  else if( iou_match_rate >= required_match_rate )
    association_regime = "iou";
  else if( motion_match_rate >= required_match_rate )
    association_regime = "motion";
  else if( registration_coverage >= min_registration_coverage &&
           registration_match_rate >= required_match_rate &&
           registration_match_rate - motion_match_rate >= registration_margin )
    association_regime = median_registered_residual <= fixed_residual ?
      "fixed" : "registration";
  else
    association_regime = "appearance";
}


void
tracking_data_statistics::compute_summary(
  size_t short_thresh, size_t long_thresh,
  double stationary_vel_thresh, double fast_vel_thresh,
  size_t crowded_thresh, size_t sparse_thresh,
  double small_area_thresh, double large_area_thresh,
  double close_distance_thresh, double high_variance_thresh )
{
  // -------------------------------------------------------------------------
  // Track length statistics
  if( !track_lengths.empty() )
  {
    std::vector< size_t > sorted_lengths = track_lengths;
    std::sort( sorted_lengths.begin(), sorted_lengths.end() );

    min_track_length = static_cast< double >( sorted_lengths.front() );
    max_track_length = static_cast< double >( sorted_lengths.back() );
    median_track_length = static_cast< double >( sorted_lengths[ sorted_lengths.size() / 2 ] );

    double sum = 0;
    for( size_t len : track_lengths )
    {
      sum += len;
      if( len < short_thresh )
        short_track_count++;
      else if( len >= long_thresh )
        long_track_count++;
      else
        medium_track_count++;
    }
    mean_track_length = sum / track_lengths.size();
  }

  // -------------------------------------------------------------------------
  // Fragmentation statistics
  if( !gaps_per_track.empty() )
  {
    double sum = 0;
    for( size_t gaps : gaps_per_track )
    {
      sum += gaps;
      if( gaps > 0 ) tracks_with_gaps++;
    }
    mean_gaps_per_track = sum / gaps_per_track.size();
    fragmentation_rate = static_cast< double >( tracks_with_gaps ) / gaps_per_track.size();
  }

  if( !gap_lengths.empty() )
  {
    mean_gap_length = std::accumulate( gap_lengths.begin(), gap_lengths.end(), 0.0 ) / gap_lengths.size();
  }

  // -------------------------------------------------------------------------
  // Motion statistics
  if( !velocities.empty() )
  {
    double sum = 0;
    double max_val = 0;
    for( double v : velocities )
    {
      sum += v;
      if( v > max_val ) max_val = v;
    }
    mean_velocity = sum / velocities.size();
    max_velocity = max_val;

    // Compute std
    double var_sum = 0;
    for( double v : velocities )
    {
      var_sum += ( v - mean_velocity ) * ( v - mean_velocity );
    }
    velocity_std = std::sqrt( var_sum / velocities.size() );

    // Classify tracks by velocity
    for( double v : velocities )
    {
      if( v < stationary_vel_thresh )
        stationary_track_count++;
      else if( v >= fast_vel_thresh )
        fast_track_count++;
      else
        slow_track_count++;
    }
  }

  if( !direction_changes.empty() )
  {
    mean_direction_change = std::accumulate(
      direction_changes.begin(), direction_changes.end(), 0.0 ) / direction_changes.size();
  }

  // -------------------------------------------------------------------------
  // Concurrent tracks
  if( !concurrent_tracks_per_frame.empty() )
  {
    double sum = 0;
    size_t max_val = 0;
    for( size_t count : concurrent_tracks_per_frame )
    {
      sum += count;
      if( count > max_val ) max_val = count;
      if( count >= crowded_thresh ) crowded_frame_count++;
      if( count > 0 && count <= sparse_thresh ) sparse_frame_count++;
    }
    mean_concurrent_tracks = sum / concurrent_tracks_per_frame.size();
    max_concurrent_tracks = static_cast< double >( max_val );
  }

  // -------------------------------------------------------------------------
  // Object size statistics
  if( !object_areas.empty() )
  {
    std::vector< double > sorted_areas = object_areas;
    std::sort( sorted_areas.begin(), sorted_areas.end() );

    min_object_area = sorted_areas.front();
    max_object_area = sorted_areas.back();
    mean_object_area = std::accumulate( object_areas.begin(), object_areas.end(), 0.0 ) / object_areas.size();

    for( double area : object_areas )
    {
      if( area < small_area_thresh )
        small_object_count++;
      else if( area >= large_area_thresh )
        large_object_count++;
      else
        medium_object_count++;
    }
  }

  // -------------------------------------------------------------------------
  // Appearance consistency
  if( !within_track_size_variance.empty() )
  {
    mean_size_variance = std::accumulate(
      within_track_size_variance.begin(), within_track_size_variance.end(), 0.0 ) /
      within_track_size_variance.size();

    size_t high_var_count = 0;
    for( double var : within_track_size_variance )
    {
      if( var > high_variance_thresh ) high_var_count++;
    }
    high_variance_track_fraction =
      static_cast< double >( high_var_count ) / within_track_size_variance.size();
  }

  // -------------------------------------------------------------------------
  // Occlusion statistics
  if( total_tracks > 0 )
  {
    occlusion_prone_fraction =
      static_cast< double >( close_track_pair_count ) / total_tracks;
  }

  // ID switch rate
  if( total_frames > 0 )
  {
    id_switch_rate = static_cast< double >( potential_id_switch_count ) / total_frames;
  }
}


void
tracking_data_statistics::log_statistics( kv::logger_handle_t logger ) const
{
  LOG_INFO( logger, "=== Tracking Data Statistics ===" );
  LOG_INFO( logger, "Total tracks: " << total_tracks
            << " (train: " << total_train_tracks
            << ", test: " << total_test_tracks << ")" );
  LOG_INFO( logger, "Total detections: " << total_detections );
  LOG_INFO( logger, "Frames: total=" << total_frames
            << ", with_tracks=" << frames_with_tracks );

  LOG_INFO( logger, "Per-class track counts:" );
  for( const auto& kv : class_track_counts )
  {
    LOG_INFO( logger, "  " << kv.first << ": " << kv.second << " tracks" );
  }

  LOG_INFO( logger, "--- Track Lengths ---" );
  LOG_INFO( logger, "  Min: " << min_track_length << ", Max: " << max_track_length
            << ", Mean: " << mean_track_length << ", Median: " << median_track_length );
  LOG_INFO( logger, "  Distribution: short=" << short_track_count
            << ", medium=" << medium_track_count << ", long=" << long_track_count );

  LOG_INFO( logger, "--- Fragmentation ---" );
  LOG_INFO( logger, "  Tracks with gaps: " << tracks_with_gaps
            << " (" << ( fragmentation_rate * 100 ) << "%)" );
  LOG_INFO( logger, "  Mean gaps/track: " << mean_gaps_per_track
            << ", Mean gap length: " << mean_gap_length );

  LOG_INFO( logger, "--- Motion ---" );
  LOG_INFO( logger, "  Velocity: mean=" << mean_velocity << ", max=" << max_velocity
            << ", std=" << velocity_std );
  LOG_INFO( logger, "  Distribution: stationary=" << stationary_track_count
            << ", slow=" << slow_track_count << ", fast=" << fast_track_count );
  LOG_INFO( logger, "  Mean direction change: " << mean_direction_change << " rad" );

  LOG_INFO( logger, "--- Concurrent Tracks ---" );
  LOG_INFO( logger, "  Mean: " << mean_concurrent_tracks << ", Max: " << max_concurrent_tracks );
  LOG_INFO( logger, "  Crowded frames: " << crowded_frame_count
            << ", Sparse frames: " << sparse_frame_count );

  LOG_INFO( logger, "--- Object Sizes ---" );
  LOG_INFO( logger, "  Area: min=" << min_object_area << ", max=" << max_object_area
            << ", mean=" << mean_object_area );
  LOG_INFO( logger, "  Distribution: small=" << small_object_count
            << ", medium=" << medium_object_count << ", large=" << large_object_count );

  LOG_INFO( logger, "--- Appearance Consistency ---" );
  LOG_INFO( logger, "  Mean within-track size variance: " << mean_size_variance );
  LOG_INFO( logger, "  High variance tracks: " << ( high_variance_track_fraction * 100 ) << "%" );

  LOG_INFO( logger, "--- Occlusion/Proximity ---" );
  LOG_INFO( logger, "  Close track pairs: " << close_track_pair_count );
  LOG_INFO( logger, "  Occlusion-prone fraction: " << ( occlusion_prone_fraction * 100 ) << "%" );
  LOG_INFO( logger, "  Potential ID switches: " << potential_id_switch_count
            << " (rate: " << id_switch_rate << "/frame)" );

  LOG_INFO( logger, "--- Association ---" );
  LOG_INFO( logger, "  Box pairs: " << association_pair_count
            << ", registered: " << registered_ious.size()
            << " (" << ( registration_coverage * 100 ) << "%)" );
  LOG_INFO( logger, "  Median IoU: raw=" << median_raw_iou
            << ", predicted=" << median_predicted_iou
            << ", registered=" << median_registered_iou );
  LOG_INFO( logger, "  Match rate: raw=" << iou_match_rate
            << ", predicted=" << motion_match_rate
            << ", registered=" << registration_match_rate );
  LOG_INFO( logger, "  Median registered residual: " << median_registered_residual
            << " box sizes" );
  LOG_INFO( logger, "  Camera motion fraction: " << camera_motion_fraction
            << ", ambiguity rate: " << ambiguity_rate );
  LOG_INFO( logger, "  Regime: " << ( association_regime.empty() ?
            "unknown" : association_regime ) );
}


// =============================================================================
// Per-trainer configuration
struct tracker_trainer_config
{
  std::string name;
  kv::algo::train_tracker_sptr trainer;

  // -------------------------------------------------------------------------
  // Hard requirements (must be met to run)
  size_t required_min_tracks = 0;
  size_t required_min_track_length = 0;
  double required_min_mean_track_length = 0;
  double required_max_fragmentation_rate = 0;    // 0 = no requirement
  double required_max_velocity = 0;              // 0 = no requirement
  size_t required_max_concurrent_tracks = 0;     // 0 = no requirement
  double required_min_object_area = 0;           // For Re-ID crop decisions

  // -------------------------------------------------------------------------
  // Soft preferences (for ranking)
  std::string track_length_preference;       // "short", "medium", "long", or empty
  std::string motion_preference;             // "stationary", "slow", "fast", or empty
  std::string density_preference;            // "sparse", "medium", "dense", or empty
  std::string fragmentation_preference;      // "continuous", "fragmented", or empty
  std::string occlusion_preference;          // "low", "medium", "high", or empty
  std::string appearance_preference;         // "consistent", "varying", or empty
  bool prefers_reid = false;                 // Prefers scenarios needing appearance features
  std::set< std::string > association_preference;  // regimes this tracker handles

  // Computed ranking
  bool association_match = false;
  double preference_score = 0.0;
};


// =============================================================================
class adaptive_tracker_trainer::priv
{
public:
  priv( adaptive_tracker_trainer& ) {}
  ~priv() {}

  // -------------------------------------------------------------------------
  // Configured trainers
  std::vector< tracker_trainer_config > m_trainers;

  // Cached training data
  kv::category_hierarchy_sptr m_labels;
  std::vector< std::string > m_train_image_names;
  std::vector< kv::object_track_set_sptr > m_train_groundtruth;
  std::vector< std::string > m_test_image_names;
  std::vector< kv::object_track_set_sptr > m_test_groundtruth;

  // Memory-based data cache
  std::vector< kv::image_container_sptr > m_train_images;
  std::vector< kv::image_container_sptr > m_test_images;
  bool m_data_from_memory = false;

  // Computed statistics
  tracking_data_statistics m_stats;

  // Logging
  kv::logger_handle_t m_logger;

  // Config parameter references (set by owner)
  size_t short_track_threshold = 10;
  size_t long_track_threshold = 100;
  double stationary_velocity_threshold = 2.0;
  double fast_velocity_threshold = 50.0;
  size_t sparse_frame_threshold = 3;
  size_t crowded_frame_threshold = 15;
  double small_object_threshold = 1024.0;
  double large_object_threshold = 16384.0;
  double close_distance_threshold = 50.0;
  double high_variance_threshold = 0.3;
  double association_match_iou = 0.3;
  double association_match_rate = 0.9;
  double max_ambiguity_rate = 0.05;
  size_t registration_min_tracks = 2;
  double registration_min_coverage = 0.5;
  double registration_margin = 0.2;
  double fixed_residual = 0.25;
  const size_t affine_min_tracks = 4;
  std::string output_statistics_file;
  bool verbose = true;
  size_t max_trainers_to_run = 3;

  // -------------------------------------------------------------------------
  // Helper methods
  void compute_statistics_from_groundtruth(
    const std::vector< kv::object_track_set_sptr >& train_gt,
    const std::vector< kv::object_track_set_sptr >& test_gt );

  void accumulate_association_statistics(
    const kv::object_track_set_sptr& track_set );

  bool check_hard_requirements( const tracker_trainer_config& tc ) const;

  double compute_preference_score( const tracker_trainer_config& tc ) const;

  std::vector< tracker_trainer_config* > select_trainers();

  void write_statistics_file() const;
};


void
adaptive_tracker_trainer::priv::compute_statistics_from_groundtruth(
  const std::vector< kv::object_track_set_sptr >& train_gt,
  const std::vector< kv::object_track_set_sptr >& test_gt )
{
  m_stats = tracking_data_statistics();  // Reset

  // Track frame ranges for concurrent track analysis
  std::map< kv::frame_id_t, size_t > tracks_per_frame;

  // Process a single track set (could be a video sequence)
  auto process_track_set = [&](
    const kv::object_track_set_sptr& track_set,
    bool is_train )
  {
    if( !track_set )
    {
      return;
    }

    auto tracks = track_set->tracks();

    for( const auto& track : tracks )
    {
      if( !track || track->empty() )
      {
        continue;
      }

      if( is_train )
        m_stats.total_train_tracks++;
      else
        m_stats.total_test_tracks++;

      m_stats.total_tracks++;

      // Get track length
      size_t track_length = track->size();
      m_stats.track_lengths.push_back( track_length );
      m_stats.total_detections += track_length;

      // Get class label from first detection with a type
      std::string track_class;
      for( const auto& state : *track )
      {
        auto obj_state = std::dynamic_pointer_cast< kv::object_track_state >( state );
        if( obj_state && obj_state->detection() && obj_state->detection()->type() )
        {
          obj_state->detection()->type()->get_most_likely( track_class );
          if( !track_class.empty() )
          {
            break;
          }
        }
      }

      if( !track_class.empty() )
      {
        m_stats.class_track_counts[ track_class ]++;
        m_stats.class_detection_counts[ track_class ] += track_length;
      }

      // Analyze track for gaps, motion, and sizes
      kv::frame_id_t prev_frame = -1;
      double prev_x = 0, prev_y = 0;
      bool has_prev = false;
      size_t gap_count = 0;
      std::vector< double > track_velocities;
      std::vector< double > track_areas;
      double prev_dx = 0, prev_dy = 0;
      bool has_prev_vel = false;

      for( const auto& state : *track )
      {
        auto obj_state = std::dynamic_pointer_cast< kv::object_track_state >( state );
        if( !obj_state )
        {
          continue;
        }

        kv::frame_id_t frame = obj_state->frame();
        tracks_per_frame[ frame ]++;
        m_stats.total_frames = std::max( m_stats.total_frames,
                                         static_cast< size_t >( frame + 1 ) );

        auto detection = obj_state->detection();
        if( !detection )
        {
          continue;
        }

        const kv::bounding_box_d& bbox = detection->bounding_box();
        double cx = ( bbox.min_x() + bbox.max_x() ) / 2.0;
        double cy = ( bbox.min_y() + bbox.max_y() ) / 2.0;
        double area = bbox.width() * bbox.height();

        m_stats.object_areas.push_back( area );
        m_stats.object_widths.push_back( bbox.width() );
        m_stats.object_heights.push_back( bbox.height() );
        track_areas.push_back( area );

        if( has_prev )
        {
          // Check for gaps
          kv::frame_id_t frame_diff = frame - prev_frame;
          if( frame_diff > 1 )
          {
            gap_count++;
            m_stats.gap_lengths.push_back( static_cast< size_t >( frame_diff - 1 ) );
          }

          // Compute velocity
          double dx = cx - prev_x;
          double dy = cy - prev_y;
          double dist = std::sqrt( dx * dx + dy * dy );
          double velocity = dist / static_cast< double >( frame_diff );
          track_velocities.push_back( velocity );
          m_stats.velocities.push_back( velocity );

          // Compute direction change
          if( has_prev_vel && ( prev_dx != 0 || prev_dy != 0 ) && ( dx != 0 || dy != 0 ) )
          {
            double dot = prev_dx * dx + prev_dy * dy;
            double mag1 = std::sqrt( prev_dx * prev_dx + prev_dy * prev_dy );
            double mag2 = std::sqrt( dx * dx + dy * dy );
            if( mag1 > 0 && mag2 > 0 )
            {
              double cos_angle = std::max( -1.0, std::min( 1.0, dot / ( mag1 * mag2 ) ) );
              double angle = std::acos( cos_angle );
              m_stats.direction_changes.push_back( angle );
            }
          }

          prev_dx = dx;
          prev_dy = dy;
          has_prev_vel = true;
        }

        prev_frame = frame;
        prev_x = cx;
        prev_y = cy;
        has_prev = true;
      }

      m_stats.gaps_per_track.push_back( gap_count );

      // Compute within-track size variance
      if( track_areas.size() > 1 )
      {
        double mean_area = std::accumulate( track_areas.begin(), track_areas.end(), 0.0 ) /
                           track_areas.size();
        double var_sum = 0;
        for( double a : track_areas )
        {
          var_sum += ( a - mean_area ) * ( a - mean_area );
        }
        double variance = var_sum / track_areas.size();
        // Normalize by mean squared for coefficient of variation
        double cv = ( mean_area > 0 ) ? std::sqrt( variance ) / mean_area : 0;
        m_stats.within_track_size_variance.push_back( cv );
      }
    }
  };

  // Process all track sets
  for( const auto& track_set : train_gt )
  {
    process_track_set( track_set, true );
  }
  for( const auto& track_set : test_gt )
  {
    process_track_set( track_set, false );
  }

  // Convert tracks_per_frame to vector
  for( const auto& kv : tracks_per_frame )
  {
    m_stats.concurrent_tracks_per_frame.push_back( kv.second );
    if( kv.second > 0 )
    {
      m_stats.frames_with_tracks++;
    }
  }

  // Analyze track proximity for occlusion detection
  // (Simplified: count frames where multiple tracks are very close)
  for( const auto& track_set : train_gt )
  {
    if( !track_set )
      continue;

    auto tracks = track_set->tracks();

    // Build frame-to-positions map
    std::map< kv::frame_id_t, std::vector< std::pair< double, double > > > frame_positions;

    for( const auto& track : tracks )
    {
      if( !track )
        continue;

      for( const auto& state : *track )
      {
        auto obj_state = std::dynamic_pointer_cast< kv::object_track_state >( state );
        if( !obj_state || !obj_state->detection() )
          continue;

        const kv::bounding_box_d& bbox = obj_state->detection()->bounding_box();
        double cx = ( bbox.min_x() + bbox.max_x() ) / 2.0;
        double cy = ( bbox.min_y() + bbox.max_y() ) / 2.0;
        frame_positions[ obj_state->frame() ].push_back( { cx, cy } );
      }
    }

    // Check for close tracks per frame
    std::set< size_t > tracks_with_close_encounters;
    for( const auto& fp : frame_positions )
    {
      const auto& positions = fp.second;
      for( size_t i = 0; i < positions.size(); ++i )
      {
        for( size_t j = i + 1; j < positions.size(); ++j )
        {
          double dx = positions[i].first - positions[j].first;
          double dy = positions[i].second - positions[j].second;
          double dist = std::sqrt( dx * dx + dy * dy );
          if( dist < close_distance_threshold )
          {
            m_stats.potential_id_switch_count++;
            m_stats.close_track_pair_count++;
          }
        }
      }
    }
  }

  for( const auto& track_set : train_gt )
  {
    accumulate_association_statistics( track_set );
  }
  for( const auto& track_set : test_gt )
  {
    accumulate_association_statistics( track_set );
  }

  // Compute summary statistics
  m_stats.compute_summary(
    short_track_threshold, long_track_threshold,
    stationary_velocity_threshold, fast_velocity_threshold,
    crowded_frame_threshold, sparse_frame_threshold,
    small_object_threshold, large_object_threshold,
    close_distance_threshold, high_variance_threshold );

  m_stats.classify_association(
    association_match_iou, association_match_rate,
    max_ambiguity_rate, registration_min_coverage,
    registration_margin, fixed_residual );

  if( verbose )
  {
    m_stats.log_statistics( m_logger );
  }
}


void
adaptive_tracker_trainer::priv::accumulate_association_statistics(
  const kv::object_track_set_sptr& track_set )
{
  if( !track_set )
  {
    return;
  }

  struct transition
  {
    kv::track_id_t id;
    kv::bounding_box_d prev, cur, predicted;
    double dx, dy;
  };

  std::map< kv::frame_id_t, std::vector< std::pair< kv::track_id_t, kv::bounding_box_d > > >
    frame_boxes;
  std::map< std::pair< kv::frame_id_t, kv::frame_id_t >, std::vector< transition > >
    transitions;

  for( const auto& track : track_set->tracks() )
  {
    if( !track )
    {
      continue;
    }

    std::vector< std::pair< kv::frame_id_t, kv::bounding_box_d > > states;
    for( const auto& state : *track )
    {
      auto obj_state = std::dynamic_pointer_cast< kv::object_track_state >( state );
      if( obj_state && obj_state->detection() &&
          obj_state->detection()->bounding_box().is_valid() )
      {
        states.emplace_back( obj_state->frame(), obj_state->detection()->bounding_box() );
      }
    }

    for( size_t i = 0; i < states.size(); ++i )
    {
      frame_boxes[ states[i].first ].emplace_back( track->id(), states[i].second );

      if( i == 0 )
      {
        continue;
      }

      const auto& prev = states[i - 1];
      const auto& cur = states[i];
      double dt = static_cast< double >( cur.first - prev.first );

      transition t{ track->id(), prev.second, cur.second, prev.second, 0, 0 };
      t.dx = cur.second.center()[0] - prev.second.center()[0];
      t.dy = cur.second.center()[1] - prev.second.center()[1];

      if( i >= 2 )
      {
        const auto& prior = states[i - 2];
        double prior_dt = static_cast< double >( prev.first - prior.first );
        double vx = ( prev.second.center()[0] - prior.second.center()[0] ) / prior_dt;
        double vy = ( prev.second.center()[1] - prior.second.center()[1] ) / prior_dt;
        t.predicted = shifted_box( prev.second, vx * dt, vy * dt );
      }

      transitions[ { prev.first, cur.first } ].push_back( t );
    }
  }

  for( const auto& group : transitions )
  {
    const auto& members = group.second;
    bool registered = members.size() >= registration_min_tracks;

    // Each box is mapped by a transform fit to the OTHER tracks only; fitting
    // on itself would register any box perfectly once there are few tracks.
    kv::vector_2d origin( 0, 0 );
    kv::matrix_3x3d normal = kv::matrix_3x3d::Zero();
    kv::vector_3d rhs_x = kv::vector_3d::Zero(), rhs_y = kv::vector_3d::Zero();

    auto design = [&]( const transition& t )
    {
      auto c = t.prev.center();
      return kv::vector_3d( c[0] - origin[0], c[1] - origin[1], 1.0 );
    };

    if( registered )
    {
      for( const auto& t : members )
      {
        origin += kv::vector_2d( t.prev.center()[0], t.prev.center()[1] );
      }
      origin /= static_cast< double >( members.size() );

      for( const auto& t : members )
      {
        kv::vector_3d p = design( t );
        normal += p * p.transpose();
        rhs_x += p * t.cur.center()[0];
        rhs_y += p * t.cur.center()[1];
      }
    }

    const auto& others = frame_boxes[ group.first.second ];

    for( const auto& t : members )
    {
      double predicted_iou = box_iou( t.predicted, t.cur );

      m_stats.association_pair_count++;
      m_stats.raw_ious.push_back( box_iou( t.prev, t.cur ) );
      m_stats.predicted_ious.push_back( predicted_iou );

      if( registered )
      {
        auto prev_c = t.prev.center();
        double mapped_x = 0, mapped_y = 0, scale = 1.0;
        bool fitted = false;

        if( members.size() - 1 >= affine_min_tracks )
        {
          kv::vector_3d p = design( t );
          kv::matrix_3x3d n = normal - p * p.transpose();
          affine_lu lu( n );
          lu.setThreshold( 1e-9 );

          if( lu.rank() == 3 )
          {
            kv::vector_3d ax = lu.solve( rhs_x - p * t.cur.center()[0] );
            kv::vector_3d ay = lu.solve( rhs_y - p * t.cur.center()[1] );
            mapped_x = ax.dot( p );
            mapped_y = ay.dot( p );
            scale = std::sqrt( std::abs( ax[0] * ay[1] - ax[1] * ay[0] ) );
            fitted = true;
          }
        }

        if( !fitted )
        {
          std::vector< double > xs, ys;
          for( const auto& o : members )
          {
            if( o.id != t.id )
            {
              xs.push_back( o.dx );
              ys.push_back( o.dy );
            }
          }
          mapped_x = prev_c[0] + median_of( xs );
          mapped_y = prev_c[1] + median_of( ys );
        }

        double hw = 0.5 * t.prev.width() * scale;
        double hh = 0.5 * t.prev.height() * scale;
        kv::bounding_box_d mapped(
          mapped_x - hw, mapped_y - hh, mapped_x + hw, mapped_y + hh );

        m_stats.registered_ious.push_back( box_iou( mapped, t.cur ) );

        double rx = t.cur.center()[0] - mapped_x;
        double ry = t.cur.center()[1] - mapped_y;
        double size = std::sqrt( std::max( t.prev.area(), 1.0 ) );
        m_stats.registered_residuals.push_back( std::sqrt( rx * rx + ry * ry ) / size );

        double total = t.dx * t.dx + t.dy * t.dy;
        m_stats.total_motion_energy += total;
        m_stats.camera_motion_energy += total - ( rx * rx + ry * ry );
      }

      for( const auto& other : others )
      {
        if( other.first == t.id )
        {
          continue;
        }

        double other_iou = box_iou( t.predicted, other.second );
        if( other_iou > 0 && other_iou >= predicted_iou )
        {
          m_stats.ambiguous_pair_count++;
          break;
        }
      }
    }
  }
}


bool
adaptive_tracker_trainer::priv::check_hard_requirements(
  const tracker_trainer_config& tc ) const
{
  // Check minimum track count
  if( tc.required_min_tracks > 0 && m_stats.total_tracks < tc.required_min_tracks )
  {
    if( verbose )
    {
      LOG_INFO( m_logger, "Trainer " << tc.name << " failed: "
                 << m_stats.total_tracks << " tracks < required "
                 << tc.required_min_tracks );
    }
    return false;
  }

  // Check minimum track length
  if( tc.required_min_track_length > 0 )
  {
    for( size_t len : m_stats.track_lengths )
    {
      if( len < tc.required_min_track_length )
      {
        if( verbose )
        {
          LOG_INFO( m_logger, "Trainer " << tc.name << " failed: "
                     << "track with length " << len << " < required "
                     << tc.required_min_track_length );
        }
        return false;
      }
    }
  }

  // Check mean track length
  if( tc.required_min_mean_track_length > 0 &&
      m_stats.mean_track_length < tc.required_min_mean_track_length )
  {
    if( verbose )
    {
      LOG_INFO( m_logger, "Trainer " << tc.name << " failed: "
                 << "mean track length " << m_stats.mean_track_length
                 << " < required " << tc.required_min_mean_track_length );
    }
    return false;
  }

  // Check fragmentation rate
  if( tc.required_max_fragmentation_rate > 0 &&
      m_stats.fragmentation_rate > tc.required_max_fragmentation_rate )
  {
    if( verbose )
    {
      LOG_INFO( m_logger, "Trainer " << tc.name << " failed: "
                 << "fragmentation rate " << m_stats.fragmentation_rate
                 << " > max " << tc.required_max_fragmentation_rate );
    }
    return false;
  }

  // Check max velocity
  if( tc.required_max_velocity > 0 && m_stats.max_velocity > tc.required_max_velocity )
  {
    if( verbose )
    {
      LOG_INFO( m_logger, "Trainer " << tc.name << " failed: "
                 << "max velocity " << m_stats.max_velocity
                 << " > limit " << tc.required_max_velocity );
    }
    return false;
  }

  // Check max concurrent tracks
  if( tc.required_max_concurrent_tracks > 0 &&
      m_stats.max_concurrent_tracks > tc.required_max_concurrent_tracks )
  {
    if( verbose )
    {
      LOG_INFO( m_logger, "Trainer " << tc.name << " failed: "
                 << "max concurrent tracks " << m_stats.max_concurrent_tracks
                 << " > limit " << tc.required_max_concurrent_tracks );
    }
    return false;
  }

  // Check min object area (for Re-ID)
  if( tc.required_min_object_area > 0 && m_stats.mean_object_area < tc.required_min_object_area )
  {
    if( verbose )
    {
      LOG_INFO( m_logger, "Trainer " << tc.name << " failed: "
                 << "mean object area " << m_stats.mean_object_area
                 << " < required " << tc.required_min_object_area );
    }
    return false;
  }

  return true;
}


double
adaptive_tracker_trainer::priv::compute_preference_score(
  const tracker_trainer_config& tc ) const
{
  double score = 0.0;

  // -------------------------------------------------------------------------
  // Track length preference
  if( !tc.track_length_preference.empty() )
  {
    double short_frac = static_cast< double >( m_stats.short_track_count ) /
                        std::max( size_t(1), m_stats.total_tracks );
    double long_frac = static_cast< double >( m_stats.long_track_count ) /
                       std::max( size_t(1), m_stats.total_tracks );

    if( tc.track_length_preference == "short" && short_frac > 0.5 )
      score += 1.0;
    else if( tc.track_length_preference == "long" && long_frac > 0.3 )
      score += 1.0;
    else if( tc.track_length_preference == "medium" &&
             short_frac <= 0.5 && long_frac <= 0.3 )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Motion preference
  if( !tc.motion_preference.empty() )
  {
    double stationary_frac = static_cast< double >( m_stats.stationary_track_count ) /
                             std::max( size_t(1), m_stats.velocities.size() );
    double fast_frac = static_cast< double >( m_stats.fast_track_count ) /
                       std::max( size_t(1), m_stats.velocities.size() );

    if( tc.motion_preference == "stationary" && stationary_frac > 0.5 )
      score += 1.0;
    else if( tc.motion_preference == "fast" && fast_frac > 0.3 )
      score += 1.0;
    else if( tc.motion_preference == "slow" &&
             stationary_frac <= 0.5 && fast_frac <= 0.3 )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Density preference
  if( !tc.density_preference.empty() )
  {
    if( tc.density_preference == "sparse" && m_stats.mean_concurrent_tracks < sparse_frame_threshold )
      score += 1.0;
    else if( tc.density_preference == "dense" && m_stats.mean_concurrent_tracks >= crowded_frame_threshold )
      score += 1.0;
    else if( tc.density_preference == "medium" &&
             m_stats.mean_concurrent_tracks >= sparse_frame_threshold &&
             m_stats.mean_concurrent_tracks < crowded_frame_threshold )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Fragmentation preference
  if( !tc.fragmentation_preference.empty() )
  {
    if( tc.fragmentation_preference == "continuous" && m_stats.fragmentation_rate < 0.1 )
      score += 1.0;
    else if( tc.fragmentation_preference == "fragmented" && m_stats.fragmentation_rate >= 0.3 )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Occlusion preference
  if( !tc.occlusion_preference.empty() )
  {
    if( tc.occlusion_preference == "low" && m_stats.occlusion_prone_fraction < 0.1 )
      score += 1.0;
    else if( tc.occlusion_preference == "high" && m_stats.occlusion_prone_fraction >= 0.3 )
      score += 1.0;
    else if( tc.occlusion_preference == "medium" &&
             m_stats.occlusion_prone_fraction >= 0.1 &&
             m_stats.occlusion_prone_fraction < 0.3 )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Appearance preference
  if( !tc.appearance_preference.empty() )
  {
    if( tc.appearance_preference == "consistent" && m_stats.high_variance_track_fraction < 0.2 )
      score += 1.0;
    else if( tc.appearance_preference == "varying" && m_stats.high_variance_track_fraction >= 0.4 )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Re-ID preference (when appearance features matter)
  if( tc.prefers_reid )
  {
    // Re-ID is beneficial when:
    // - There are many concurrent tracks (need to distinguish)
    // - Objects are large enough for good crops
    // - There's occlusion (need appearance to recover)
    bool reid_beneficial =
      m_stats.mean_concurrent_tracks >= 5 ||
      m_stats.occlusion_prone_fraction >= 0.2 ||
      m_stats.mean_object_area >= small_object_threshold;

    if( reid_beneficial )
      score += 1.0;
  }

  return score;
}


std::vector< tracker_trainer_config* >
adaptive_tracker_trainer::priv::select_trainers()
{
  std::vector< tracker_trainer_config* > qualifying;

  for( auto& tc : m_trainers )
  {
    if( check_hard_requirements( tc ) )
    {
      tc.preference_score = compute_preference_score( tc );
      tc.association_match =
        tc.association_preference.count( m_stats.association_regime ) > 0;
      qualifying.push_back( &tc );

      if( verbose )
      {
        LOG_INFO( m_logger, "Trainer " << tc.name << " qualifies with preference score "
                  << tc.preference_score << ( tc.association_match ?
                  ", matches association regime" : "" ) );
      }
    }
    else
    {
      if( verbose )
      {
        LOG_INFO( m_logger, "Trainer " << tc.name << " does not meet hard requirements" );
      }
    }
  }

  // The association regime decides; preferences only order within it
  std::stable_sort( qualifying.begin(), qualifying.end(),
    []( const tracker_trainer_config* a, const tracker_trainer_config* b )
    {
      if( a->association_match != b->association_match )
      {
        return a->association_match;
      }
      return a->preference_score > b->preference_score;
    } );

  return qualifying;
}


void
adaptive_tracker_trainer::priv::write_statistics_file() const
{
  std::ofstream out( output_statistics_file );
  if( !out.is_open() )
  {
    LOG_WARN( m_logger, "Could not open statistics file: " << output_statistics_file );
    return;
  }

  out << std::fixed << std::setprecision( 4 );

  out << "{\n";

  // Track counts
  out << "  \"track_counts\": {\n";
  out << "    \"total\": " << m_stats.total_tracks << ",\n";
  out << "    \"train\": " << m_stats.total_train_tracks << ",\n";
  out << "    \"test\": " << m_stats.total_test_tracks << ",\n";
  out << "    \"total_detections\": " << m_stats.total_detections << "\n";
  out << "  },\n";

  // Class counts
  out << "  \"class_track_counts\": {\n";
  bool first = true;
  for( const auto& kv : m_stats.class_track_counts )
  {
    if( !first ) out << ",\n";
    out << "    \"" << kv.first << "\": " << kv.second;
    first = false;
  }
  out << "\n  },\n";

  // Track lengths
  out << "  \"track_lengths\": {\n";
  out << "    \"min\": " << m_stats.min_track_length << ",\n";
  out << "    \"max\": " << m_stats.max_track_length << ",\n";
  out << "    \"mean\": " << m_stats.mean_track_length << ",\n";
  out << "    \"median\": " << m_stats.median_track_length << ",\n";
  out << "    \"distribution\": { \"short\": " << m_stats.short_track_count
      << ", \"medium\": " << m_stats.medium_track_count
      << ", \"long\": " << m_stats.long_track_count << " }\n";
  out << "  },\n";

  // Fragmentation
  out << "  \"fragmentation\": {\n";
  out << "    \"tracks_with_gaps\": " << m_stats.tracks_with_gaps << ",\n";
  out << "    \"fragmentation_rate\": " << m_stats.fragmentation_rate << ",\n";
  out << "    \"mean_gaps_per_track\": " << m_stats.mean_gaps_per_track << ",\n";
  out << "    \"mean_gap_length\": " << m_stats.mean_gap_length << "\n";
  out << "  },\n";

  // Motion
  out << "  \"motion\": {\n";
  out << "    \"mean_velocity\": " << m_stats.mean_velocity << ",\n";
  out << "    \"max_velocity\": " << m_stats.max_velocity << ",\n";
  out << "    \"velocity_std\": " << m_stats.velocity_std << ",\n";
  out << "    \"mean_direction_change\": " << m_stats.mean_direction_change << ",\n";
  out << "    \"distribution\": { \"stationary\": " << m_stats.stationary_track_count
      << ", \"slow\": " << m_stats.slow_track_count
      << ", \"fast\": " << m_stats.fast_track_count << " }\n";
  out << "  },\n";

  // Concurrent tracks
  out << "  \"concurrent_tracks\": {\n";
  out << "    \"mean\": " << m_stats.mean_concurrent_tracks << ",\n";
  out << "    \"max\": " << m_stats.max_concurrent_tracks << ",\n";
  out << "    \"crowded_frames\": " << m_stats.crowded_frame_count << ",\n";
  out << "    \"sparse_frames\": " << m_stats.sparse_frame_count << "\n";
  out << "  },\n";

  // Object sizes
  out << "  \"object_sizes\": {\n";
  out << "    \"mean_area\": " << m_stats.mean_object_area << ",\n";
  out << "    \"min_area\": " << m_stats.min_object_area << ",\n";
  out << "    \"max_area\": " << m_stats.max_object_area << ",\n";
  out << "    \"distribution\": { \"small\": " << m_stats.small_object_count
      << ", \"medium\": " << m_stats.medium_object_count
      << ", \"large\": " << m_stats.large_object_count << " }\n";
  out << "  },\n";

  // Appearance
  out << "  \"appearance\": {\n";
  out << "    \"mean_size_variance\": " << m_stats.mean_size_variance << ",\n";
  out << "    \"high_variance_fraction\": " << m_stats.high_variance_track_fraction << "\n";
  out << "  },\n";

  // Occlusion
  out << "  \"occlusion\": {\n";
  out << "    \"close_track_pairs\": " << m_stats.close_track_pair_count << ",\n";
  out << "    \"occlusion_prone_fraction\": " << m_stats.occlusion_prone_fraction << ",\n";
  out << "    \"potential_id_switches\": " << m_stats.potential_id_switch_count << ",\n";
  out << "    \"id_switch_rate\": " << m_stats.id_switch_rate << "\n";
  out << "  },\n";

  // Association
  out << "  \"association\": {\n";
  out << "    \"pairs\": " << m_stats.association_pair_count << ",\n";
  out << "    \"registration_coverage\": " << m_stats.registration_coverage << ",\n";
  out << "    \"median_raw_iou\": " << m_stats.median_raw_iou << ",\n";
  out << "    \"median_predicted_iou\": " << m_stats.median_predicted_iou << ",\n";
  out << "    \"median_registered_iou\": " << m_stats.median_registered_iou << ",\n";
  out << "    \"median_registered_residual\": " << m_stats.median_registered_residual << ",\n";
  out << "    \"iou_match_rate\": " << m_stats.iou_match_rate << ",\n";
  out << "    \"motion_match_rate\": " << m_stats.motion_match_rate << ",\n";
  out << "    \"registration_match_rate\": " << m_stats.registration_match_rate << ",\n";
  out << "    \"camera_motion_fraction\": " << m_stats.camera_motion_fraction << ",\n";
  out << "    \"ambiguity_rate\": " << m_stats.ambiguity_rate << ",\n";
  out << "    \"regime\": \"" << m_stats.association_regime << "\"\n";
  out << "  },\n";

  // Frames
  out << "  \"frames\": {\n";
  out << "    \"total\": " << m_stats.total_frames << ",\n";
  out << "    \"with_tracks\": " << m_stats.frames_with_tracks << "\n";
  out << "  }\n";

  out << "}\n";

  out.close();
  LOG_INFO( m_logger, "Wrote statistics to: " << output_statistics_file );
}


// =============================================================================
void
adaptive_tracker_trainer
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  d->m_logger = this->logger();
}


// -----------------------------------------------------------------------------
kv::config_block_sptr
adaptive_tracker_trainer
::get_configuration() const
{
  // Get base config from base class (includes PLUGGABLE_IMPL params)
  kv::config_block_sptr config = kv::algo::train_tracker::get_configuration();

  // Add static params from this class
  kv::config_block_sptr cb = config;
  CPP_MAGIC_MAP( PARAM_CONFIG_GET_FROM_THIS, CPP_MAGIC_EMPTY, VIAME_CORE_ATT_PARAMS )

  // -------------------------------------------------------------------------
  // Trainer configurations
  for( size_t i = 0; i < d->m_trainers.size(); ++i )
  {
    const auto& tc = d->m_trainers[i];
    std::string prefix = "trainer_" + std::to_string( i + 1 ) + ":";

    // Hard requirements
    config->set_value( prefix + "required_min_tracks", tc.required_min_tracks,
      "Minimum total tracks. 0 = no requirement." );
    config->set_value( prefix + "required_min_track_length", tc.required_min_track_length,
      "Minimum track length (frames). 0 = no requirement." );
    config->set_value( prefix + "required_min_mean_track_length", tc.required_min_mean_track_length,
      "Minimum mean track length. 0 = no requirement." );
    config->set_value( prefix + "required_max_fragmentation_rate", tc.required_max_fragmentation_rate,
      "Max allowed fragmentation rate. 0 = no requirement." );
    config->set_value( prefix + "required_max_velocity", tc.required_max_velocity,
      "Max allowed velocity (pixels/frame). 0 = no requirement." );
    config->set_value( prefix + "required_max_concurrent_tracks", tc.required_max_concurrent_tracks,
      "Max allowed concurrent tracks. 0 = no requirement." );
    config->set_value( prefix + "required_min_object_area", tc.required_min_object_area,
      "Min mean object area for Re-ID crops. 0 = no requirement." );

    // Soft preferences
    config->set_value( prefix + "track_length_preference", tc.track_length_preference,
      "Preference: 'short', 'medium', 'long', or empty." );
    config->set_value( prefix + "motion_preference", tc.motion_preference,
      "Preference: 'stationary', 'slow', 'fast', or empty." );
    config->set_value( prefix + "density_preference", tc.density_preference,
      "Preference: 'sparse', 'medium', 'dense', or empty." );
    config->set_value( prefix + "fragmentation_preference", tc.fragmentation_preference,
      "Preference: 'continuous', 'fragmented', or empty." );
    config->set_value( prefix + "occlusion_preference", tc.occlusion_preference,
      "Preference: 'low', 'medium', 'high', or empty." );
    config->set_value( prefix + "appearance_preference", tc.appearance_preference,
      "Preference: 'consistent', 'varying', or empty." );
    config->set_value( prefix + "prefers_reid", tc.prefers_reid,
      "Prefer scenarios where Re-ID/appearance features help. Default: false" );

    std::string regimes;
    for( const auto& r : tc.association_preference )
    {
      regimes += ( regimes.empty() ? "" : "," ) + r;
    }
    config->set_value( prefix + "association_preference", regimes,
      "Comma list of association regimes this tracker handles: 'iou', "
      "'motion', 'registration', 'fixed', 'appearance'. A trainer matching the data's "
      "regime outranks every trainer that does not." );

    kv::get_nested_algo_configuration<kv::algo::train_tracker>(
      prefix + "trainer", config, tc.trainer );
  }

  return config;
}


// -----------------------------------------------------------------------------
void
adaptive_tracker_trainer
::set_configuration_internal( kv::config_block_sptr config_in )
{
  // Merge with defaults
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  // Copy config values to priv for use by helper methods
  d->max_trainers_to_run = c_max_trainers_to_run;
  d->short_track_threshold = c_short_track_threshold;
  d->long_track_threshold = c_long_track_threshold;
  d->stationary_velocity_threshold = c_stationary_velocity_threshold;
  d->fast_velocity_threshold = c_fast_velocity_threshold;
  d->sparse_frame_threshold = c_sparse_frame_threshold;
  d->crowded_frame_threshold = c_crowded_frame_threshold;
  d->small_object_threshold = c_small_object_threshold;
  d->large_object_threshold = c_large_object_threshold;
  d->close_distance_threshold = c_close_distance_threshold;
  d->high_variance_threshold = c_high_variance_threshold;
  d->association_match_iou = c_association_match_iou;
  d->association_match_rate = c_association_match_rate;
  d->max_ambiguity_rate = c_max_ambiguity_rate;
  d->registration_min_tracks = c_registration_min_tracks;
  d->registration_min_coverage = c_registration_min_coverage;
  d->registration_margin = c_registration_margin;
  d->fixed_residual = c_fixed_residual;
  d->output_statistics_file = c_output_statistics_file;
  d->verbose = c_verbose;

  // -------------------------------------------------------------------------
  // Trainer configurations
  d->m_trainers.clear();

  for( size_t i = 1; i <= 100; ++i )
  {
    std::string prefix = "trainer_" + std::to_string( i ) + ":";
    std::string trainer_key = prefix + "trainer";

    if( !config->has_value( trainer_key + ":type" ) )
    {
      break;
    }

    tracker_trainer_config tc;
    tc.name = "trainer_" + std::to_string( i );

    // Hard requirements
    tc.required_min_tracks =
      config->get_value< size_t >( prefix + "required_min_tracks", 0 );
    tc.required_min_track_length =
      config->get_value< size_t >( prefix + "required_min_track_length", 0 );
    tc.required_min_mean_track_length =
      config->get_value< double >( prefix + "required_min_mean_track_length", 0.0 );
    tc.required_max_fragmentation_rate =
      config->get_value< double >( prefix + "required_max_fragmentation_rate", 0.0 );
    tc.required_max_velocity =
      config->get_value< double >( prefix + "required_max_velocity", 0.0 );
    tc.required_max_concurrent_tracks =
      config->get_value< size_t >( prefix + "required_max_concurrent_tracks", 0 );
    tc.required_min_object_area =
      config->get_value< double >( prefix + "required_min_object_area", 0.0 );

    // Soft preferences
    tc.track_length_preference =
      config->get_value< std::string >( prefix + "track_length_preference", "" );
    tc.motion_preference =
      config->get_value< std::string >( prefix + "motion_preference", "" );
    tc.density_preference =
      config->get_value< std::string >( prefix + "density_preference", "" );
    tc.fragmentation_preference =
      config->get_value< std::string >( prefix + "fragmentation_preference", "" );
    tc.occlusion_preference =
      config->get_value< std::string >( prefix + "occlusion_preference", "" );
    tc.appearance_preference =
      config->get_value< std::string >( prefix + "appearance_preference", "" );
    tc.prefers_reid =
      config->get_value< bool >( prefix + "prefers_reid", false );

    std::stringstream regimes(
      config->get_value< std::string >( prefix + "association_preference", "" ) );
    for( std::string r; std::getline( regimes, r, ',' ); )
    {
      r.erase( 0, r.find_first_not_of( " \t" ) );
      r.erase( r.find_last_not_of( " \t" ) + 1 );
      if( !r.empty() )
      {
        tc.association_preference.insert( r );
      }
    }

    // Nested trainer
    kv::algo::train_tracker_sptr trainer;
    kv::set_nested_algo_configuration<kv::algo::train_tracker>( trainer_key, config, trainer );
    tc.trainer = trainer;

    if( tc.trainer )
    {
      d->m_trainers.push_back( tc );
      LOG_DEBUG( d->m_logger, "Loaded tracker trainer configuration: " << tc.name );
    }
    else
    {
      LOG_WARN( d->m_logger, "Failed to create tracker trainer: " << tc.name );
    }
  }

  LOG_INFO( d->m_logger, "Configured " << d->m_trainers.size() << " tracker trainers" );
}


// -----------------------------------------------------------------------------
bool
adaptive_tracker_trainer
::check_configuration( kv::config_block_sptr config ) const
{
  for( size_t i = 1; i <= 100; ++i )
  {
    std::string trainer_key = "trainer_" + std::to_string( i ) + ":trainer";
    if( config->has_value( trainer_key + ":type" ) )
    {
      if( kv::check_nested_algo_configuration<kv::algo::train_tracker>( trainer_key, config ) )
      {
        return true;
      }
    }
    else
    {
      break;
    }
  }

  LOG_ERROR( logger(), "No valid tracker trainers configured." );
  return false;
}


// -----------------------------------------------------------------------------
void
adaptive_tracker_trainer
::add_data_from_disk(
  kv::category_hierarchy_sptr object_labels,
  std::vector< std::string > train_image_names,
  std::vector< kv::object_track_set_sptr > train_groundtruth,
  std::vector< std::string > test_image_names,
  std::vector< kv::object_track_set_sptr > test_groundtruth )
{
  d->m_labels = object_labels;
  d->m_train_image_names = train_image_names;
  d->m_train_groundtruth = train_groundtruth;
  d->m_test_image_names = test_image_names;
  d->m_test_groundtruth = test_groundtruth;
  d->m_data_from_memory = false;

  LOG_INFO( d->m_logger, "Analyzing tracking data statistics..." );
  d->compute_statistics_from_groundtruth( train_groundtruth, test_groundtruth );
}


// -----------------------------------------------------------------------------
void
adaptive_tracker_trainer
::add_data_from_memory(
  kv::category_hierarchy_sptr object_labels,
  std::vector< kv::image_container_sptr > train_images,
  std::vector< kv::object_track_set_sptr > train_groundtruth,
  std::vector< kv::image_container_sptr > test_images,
  std::vector< kv::object_track_set_sptr > test_groundtruth )
{
  d->m_labels = object_labels;
  d->m_train_images = train_images;
  d->m_train_groundtruth = train_groundtruth;
  d->m_test_images = test_images;
  d->m_test_groundtruth = test_groundtruth;
  d->m_data_from_memory = true;

  LOG_INFO( d->m_logger, "Analyzing tracking data statistics..." );
  d->compute_statistics_from_groundtruth( train_groundtruth, test_groundtruth );
}


// -----------------------------------------------------------------------------
std::map<std::string, std::string>
adaptive_tracker_trainer
::update_model()
{
  std::map<std::string, std::string> combined_output;

  LOG_INFO( d->m_logger, "Selecting tracker trainers to run..." );

  // Write statistics file if configured
  if( !c_output_statistics_file.empty() )
  {
    d->write_statistics_file();
  }

  // Select qualifying trainers
  auto selected = d->select_trainers();

  if( selected.empty() )
  {
    VITAL_THROW( kv::invalid_data,
      "no tracker trainer met its hard requirements for this data: " +
      std::to_string( d->m_stats.total_tracks ) + " tracks, mean length " +
      std::to_string( d->m_stats.mean_track_length ) + ", fragmentation rate " +
      std::to_string( d->m_stats.fragmentation_rate ) + ". Each trainer's "
      "unmet requirement is logged above. Returning an empty result here made "
      "the run report success with no model." );
  }

  LOG_INFO( d->m_logger, "Running up to " << d->max_trainers_to_run
            << " of " << selected.size() << " qualifying tracker trainer(s)..." );

  // A failed trainer hands its slot to the next-ranked one
  size_t completed = 0;
  for( size_t i = 0; i < selected.size() && completed < d->max_trainers_to_run; ++i )
  {
    auto& tc = *selected[i];
    LOG_INFO( d->m_logger, "Running tracker trainer " << ( i + 1 ) << "/" << selected.size()
              << ": " << tc.name << " (score: " << tc.preference_score << ")" );

    if( !tc.trainer )
    {
      LOG_ERROR( d->m_logger, "Trainer " << tc.name << " has null implementation!" );
      continue;
    }

    try
    {
      // Pass data to nested trainer
      if( d->m_data_from_memory )
      {
        tc.trainer->add_data_from_memory(
          d->m_labels,
          d->m_train_images, d->m_train_groundtruth,
          d->m_test_images, d->m_test_groundtruth );
      }
      else
      {
        tc.trainer->add_data_from_disk(
          d->m_labels,
          d->m_train_image_names, d->m_train_groundtruth,
          d->m_test_image_names, d->m_test_groundtruth );
      }

      // Run training
      std::map<std::string, std::string> trainer_output = tc.trainer->update_model();

      // Trainers run best-first, so an earlier trainer's keys (notably
      // "type") win over a later one's
      for( const auto& pair : trainer_output )
      {
        combined_output.insert( pair );
      }

      completed++;
      LOG_INFO( d->m_logger, "Completed tracker trainer: " << tc.name );
    }
    catch( const std::exception& e )
    {
      LOG_ERROR( d->m_logger, "Tracker trainer " << tc.name << " failed: " << e.what() );
    }
  }

  if( combined_output.empty() )
  {
    VITAL_THROW( kv::invalid_data,
      "every selected tracker trainer failed; see the errors above. No model "
      "was produced." );
  }

  LOG_INFO( d->m_logger, "Adaptive tracker training complete." );

  return combined_output;
}

} // end namespace viame
