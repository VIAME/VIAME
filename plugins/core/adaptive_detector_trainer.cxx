/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "adaptive_detector_trainer.h"

#include <vital/util/cpu_timer.h>
#include <vital/types/image_container.h>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace viame {

namespace kv = kwiver::vital;

// =============================================================================
// Statistics structure for training data analysis
struct training_data_statistics
{
  // -------------------------------------------------------------------------
  // Annotation counts
  size_t total_train_annotations = 0;
  size_t total_test_annotations = 0;
  size_t train_frames_with_annotations = 0;
  size_t test_frames_with_annotations = 0;
  size_t total_frames = 0;

  // Per-class annotation counts
  std::map< std::string, size_t > class_counts;

  // -------------------------------------------------------------------------
  // Object size data (raw values for percentile computation)
  std::vector< double > all_widths;
  std::vector< double > all_heights;
  std::vector< double > all_areas;
  std::vector< double > all_aspect_ratios;

  // Size summary statistics
  double min_width = 0, max_width = 0, mean_width = 0, median_width = 0;
  double min_height = 0, max_height = 0, mean_height = 0, median_height = 0;
  double min_area = 0, max_area = 0, mean_area = 0, median_area = 0;

  // Size distribution counts
  size_t small_object_count = 0;
  size_t medium_object_count = 0;
  size_t large_object_count = 0;

  // Size percentiles
  double area_10th_percentile = 0;
  double area_50th_percentile = 0;
  double area_90th_percentile = 0;

  // -------------------------------------------------------------------------
  // Aspect ratio statistics
  double mean_aspect_ratio = 0;
  double aspect_ratio_std = 0;
  size_t tall_object_count = 0;    // aspect < 0.5
  size_t square_object_count = 0;  // 0.5 <= aspect <= 2.0
  size_t wide_object_count = 0;    // aspect > 2.0

  // -------------------------------------------------------------------------
  // Object density per image
  std::vector< size_t > objects_per_frame;
  double mean_objects_per_frame = 0;
  double max_objects_per_frame = 0;
  size_t crowded_frame_count = 0;  // frames with many objects
  size_t sparse_frame_count = 0;   // frames with few objects

  // -------------------------------------------------------------------------
  // Scale variance
  double scale_variance = 0;
  double min_max_area_ratio = 0;
  bool is_multi_scale = false;

  // -------------------------------------------------------------------------
  // Class imbalance
  double class_imbalance_ratio = 0;  // max_count / min_count
  size_t rare_class_count = 0;       // classes with < threshold samples
  size_t dominant_class_count = 0;   // classes with > threshold samples

  // -------------------------------------------------------------------------
  // Spatial distribution (per-frame accumulation)
  size_t edge_object_count = 0;      // objects near image borders
  double edge_object_fraction = 0;

  // -------------------------------------------------------------------------
  // Occlusion/overlap analysis
  double mean_inter_object_iou = 0;
  size_t overlapping_pair_count = 0;
  double high_overlap_fraction = 0;  // fraction of objects with significant overlap

  // -------------------------------------------------------------------------
  // Mask presence
  size_t objects_with_masks = 0;
  double mask_fraction = 0;
  bool has_masks = false;

  // -------------------------------------------------------------------------
  // Methods
  void compute_summary(
    double small_thresh, double large_thresh,
    double tall_thresh, double wide_thresh,
    size_t crowded_thresh, size_t sparse_thresh,
    size_t rare_class_thresh, size_t dominant_class_thresh,
    double edge_fraction_thresh, double overlap_thresh );

  void log_statistics( kv::logger_handle_t logger ) const;
};


void
training_data_statistics::compute_summary(
  double small_thresh, double large_thresh,
  double tall_thresh, double wide_thresh,
  size_t crowded_thresh, size_t sparse_thresh,
  size_t rare_class_thresh, size_t dominant_class_thresh,
  double edge_fraction_thresh, double overlap_thresh )
{
  if( all_areas.empty() )
  {
    return;
  }

  size_t n = all_areas.size();

  // -------------------------------------------------------------------------
  // Create sorted copies for percentile computation
  std::vector< double > sorted_widths = all_widths;
  std::vector< double > sorted_heights = all_heights;
  std::vector< double > sorted_areas = all_areas;

  std::sort( sorted_widths.begin(), sorted_widths.end() );
  std::sort( sorted_heights.begin(), sorted_heights.end() );
  std::sort( sorted_areas.begin(), sorted_areas.end() );

  // Width statistics
  min_width = sorted_widths.front();
  max_width = sorted_widths.back();
  median_width = sorted_widths[ n / 2 ];
  mean_width = std::accumulate( all_widths.begin(), all_widths.end(), 0.0 ) / n;

  // Height statistics
  min_height = sorted_heights.front();
  max_height = sorted_heights.back();
  median_height = sorted_heights[ n / 2 ];
  mean_height = std::accumulate( all_heights.begin(), all_heights.end(), 0.0 ) / n;

  // Area statistics
  min_area = sorted_areas.front();
  max_area = sorted_areas.back();
  median_area = sorted_areas[ n / 2 ];
  mean_area = std::accumulate( all_areas.begin(), all_areas.end(), 0.0 ) / n;

  // Size percentiles
  area_10th_percentile = sorted_areas[ static_cast< size_t >( n * 0.1 ) ];
  area_50th_percentile = sorted_areas[ n / 2 ];
  area_90th_percentile = sorted_areas[ std::min( static_cast< size_t >( n * 0.9 ), n - 1 ) ];

  // Size distribution
  for( double area : all_areas )
  {
    if( area < small_thresh )
      small_object_count++;
    else if( area >= large_thresh )
      large_object_count++;
    else
      medium_object_count++;
  }

  // -------------------------------------------------------------------------
  // Aspect ratio statistics
  if( !all_aspect_ratios.empty() )
  {
    mean_aspect_ratio = std::accumulate(
      all_aspect_ratios.begin(), all_aspect_ratios.end(), 0.0 ) / all_aspect_ratios.size();

    double variance_sum = 0;
    for( double ar : all_aspect_ratios )
    {
      variance_sum += ( ar - mean_aspect_ratio ) * ( ar - mean_aspect_ratio );
    }
    aspect_ratio_std = std::sqrt( variance_sum / all_aspect_ratios.size() );

    for( double ar : all_aspect_ratios )
    {
      if( ar < tall_thresh )
        tall_object_count++;
      else if( ar > wide_thresh )
        wide_object_count++;
      else
        square_object_count++;
    }
  }

  // -------------------------------------------------------------------------
  // Object density per frame
  if( !objects_per_frame.empty() )
  {
    double sum = 0;
    size_t max_val = 0;
    for( size_t count : objects_per_frame )
    {
      sum += count;
      if( count > max_val ) max_val = count;
      if( count >= crowded_thresh ) crowded_frame_count++;
      if( count <= sparse_thresh && count > 0 ) sparse_frame_count++;
    }
    mean_objects_per_frame = sum / objects_per_frame.size();
    max_objects_per_frame = static_cast< double >( max_val );
  }

  // -------------------------------------------------------------------------
  // Scale variance
  if( min_area > 0 )
  {
    min_max_area_ratio = max_area / min_area;
  }

  // Compute variance of sqrt(area) for scale variance
  std::vector< double > sqrt_areas;
  sqrt_areas.reserve( n );
  for( double area : all_areas )
  {
    sqrt_areas.push_back( std::sqrt( area ) );
  }
  double mean_sqrt = std::accumulate( sqrt_areas.begin(), sqrt_areas.end(), 0.0 ) / n;
  double var_sum = 0;
  for( double sa : sqrt_areas )
  {
    var_sum += ( sa - mean_sqrt ) * ( sa - mean_sqrt );
  }
  scale_variance = var_sum / n;

  // Multi-scale if objects vary by more than 10x in area
  is_multi_scale = ( min_max_area_ratio > 100 );  // sqrt gives 10x

  // -------------------------------------------------------------------------
  // Class imbalance
  if( !class_counts.empty() )
  {
    size_t min_count = std::numeric_limits< size_t >::max();
    size_t max_count = 0;

    for( const auto& kv : class_counts )
    {
      if( kv.second < min_count ) min_count = kv.second;
      if( kv.second > max_count ) max_count = kv.second;
      if( kv.second < rare_class_thresh ) rare_class_count++;
      if( kv.second > dominant_class_thresh ) dominant_class_count++;
    }

    if( min_count > 0 )
    {
      class_imbalance_ratio = static_cast< double >( max_count ) / min_count;
    }
  }

  // -------------------------------------------------------------------------
  // Edge object fraction
  if( n > 0 )
  {
    edge_object_fraction = static_cast< double >( edge_object_count ) / n;
  }

  // -------------------------------------------------------------------------
  // Mask fraction
  if( n > 0 )
  {
    mask_fraction = static_cast< double >( objects_with_masks ) / n;
    has_masks = ( mask_fraction > 0.5 );  // Majority have masks
  }

  // -------------------------------------------------------------------------
  // Overlap fraction (computed during data collection, just normalize here)
  // high_overlap_fraction is set during compute_statistics_from_groundtruth
}


void
training_data_statistics::log_statistics( kv::logger_handle_t logger ) const
{
  size_t total = total_train_annotations + total_test_annotations;

  LOG_INFO( logger, "=== Training Data Statistics ===" );
  LOG_INFO( logger, "Total annotations: " << total
            << " (train: " << total_train_annotations
            << ", test: " << total_test_annotations << ")" );
  LOG_INFO( logger, "Frames: total=" << total_frames
            << ", with_annotations=" << ( train_frames_with_annotations + test_frames_with_annotations ) );

  LOG_INFO( logger, "Per-class counts:" );
  for( const auto& kv : class_counts )
  {
    LOG_INFO( logger, "  " << kv.first << ": " << kv.second );
  }

  if( !all_areas.empty() )
  {
    LOG_INFO( logger, "--- Object Sizes ---" );
    LOG_INFO( logger, "  Area: min=" << min_area << ", max=" << max_area
              << ", mean=" << mean_area << ", median=" << median_area );
    LOG_INFO( logger, "  Percentiles: p10=" << area_10th_percentile
              << ", p50=" << area_50th_percentile << ", p90=" << area_90th_percentile );
    LOG_INFO( logger, "  Distribution: small=" << small_object_count
              << ", medium=" << medium_object_count << ", large=" << large_object_count );

    LOG_INFO( logger, "--- Aspect Ratios ---" );
    LOG_INFO( logger, "  Mean: " << mean_aspect_ratio << ", StdDev: " << aspect_ratio_std );
    LOG_INFO( logger, "  Distribution: tall=" << tall_object_count
              << ", square=" << square_object_count << ", wide=" << wide_object_count );

    LOG_INFO( logger, "--- Object Density ---" );
    LOG_INFO( logger, "  Mean objects/frame: " << mean_objects_per_frame
              << ", Max: " << max_objects_per_frame );
    LOG_INFO( logger, "  Crowded frames: " << crowded_frame_count
              << ", Sparse frames: " << sparse_frame_count );

    LOG_INFO( logger, "--- Scale Variance ---" );
    LOG_INFO( logger, "  Scale variance: " << scale_variance
              << ", Min/Max ratio: " << min_max_area_ratio );
    LOG_INFO( logger, "  Multi-scale: " << ( is_multi_scale ? "yes" : "no" ) );

    LOG_INFO( logger, "--- Class Imbalance ---" );
    LOG_INFO( logger, "  Imbalance ratio: " << class_imbalance_ratio );
    LOG_INFO( logger, "  Rare classes: " << rare_class_count
              << ", Dominant classes: " << dominant_class_count );

    LOG_INFO( logger, "--- Spatial Distribution ---" );
    LOG_INFO( logger, "  Edge objects: " << edge_object_count
              << " (" << ( edge_object_fraction * 100 ) << "%)" );

    LOG_INFO( logger, "--- Overlap/Occlusion ---" );
    LOG_INFO( logger, "  Mean inter-object IoU: " << mean_inter_object_iou );
    LOG_INFO( logger, "  Overlapping pairs: " << overlapping_pair_count
              << ", High overlap fraction: " << ( high_overlap_fraction * 100 ) << "%" );

    LOG_INFO( logger, "--- Masks ---" );
    LOG_INFO( logger, "  Objects with masks: " << objects_with_masks
              << " (" << ( mask_fraction * 100 ) << "%)" );
    LOG_INFO( logger, "  Has masks (majority): " << ( has_masks ? "yes" : "no" ) );
  }
}


// =============================================================================
// Per-trainer configuration
struct trainer_config
{
  std::string name;
  kv::algo::train_detector_sptr trainer;

  // -------------------------------------------------------------------------
  // Hard requirements (must be met to run)
  size_t required_min_count_per_class = 0;
  double required_min_object_area = 0;
  double required_percentile = 0.5;
  double required_max_aspect_ratio_std = 0;      // 0 = no requirement
  double required_max_objects_per_frame = 0;     // 0 = no requirement
  double required_max_class_imbalance = 0;       // 0 = no requirement
  bool required_masks = false;                   // true = must have masks

  // -------------------------------------------------------------------------
  // Soft preferences (for ranking)
  std::string annotation_count_preference;   // "low", "medium", "high", or empty
  std::string object_size_preference;        // "small", "medium", "large", or empty
  std::string aspect_ratio_preference;       // "tall", "square", "wide", or empty
  std::string density_preference;            // "sparse", "medium", "dense", or empty
  std::string scale_preference;              // "uniform", "multi-scale", or empty
  std::string overlap_preference;            // "low", "medium", "high", or empty
  bool prefers_masks = false;

  // Computed preference score
  double preference_score = 0.0;
};


// =============================================================================
class adaptive_detector_trainer::priv
{
public:
  priv()
    : m_max_trainers_to_run( 3 )
    // Size thresholds
    , m_small_object_threshold( 1024.0 )
    , m_large_object_threshold( 16384.0 )
    // Aspect ratio thresholds
    , m_tall_aspect_threshold( 0.5 )
    , m_wide_aspect_threshold( 2.0 )
    // Annotation count thresholds
    , m_low_annotation_threshold( 500 )
    , m_high_annotation_threshold( 2000 )
    // Density thresholds
    , m_sparse_frame_threshold( 5 )
    , m_crowded_frame_threshold( 20 )
    // Class thresholds
    , m_rare_class_threshold( 50 )
    , m_dominant_class_threshold( 500 )
    // Edge detection threshold (fraction of image dimension)
    , m_edge_margin_fraction( 0.05 )
    // Overlap threshold for "high overlap"
    , m_overlap_iou_threshold( 0.3 )
    // Output
    , m_output_statistics_file( "" )
    , m_verbose( true )
  {}

  ~priv() {}

  // -------------------------------------------------------------------------
  // Global configuration
  size_t m_max_trainers_to_run;

  // Size thresholds
  double m_small_object_threshold;
  double m_large_object_threshold;

  // Aspect ratio thresholds
  double m_tall_aspect_threshold;
  double m_wide_aspect_threshold;

  // Annotation count thresholds
  size_t m_low_annotation_threshold;
  size_t m_high_annotation_threshold;

  // Density thresholds
  size_t m_sparse_frame_threshold;
  size_t m_crowded_frame_threshold;

  // Class thresholds
  size_t m_rare_class_threshold;
  size_t m_dominant_class_threshold;

  // Edge detection
  double m_edge_margin_fraction;

  // Overlap threshold
  double m_overlap_iou_threshold;

  // Output
  std::string m_output_statistics_file;
  bool m_verbose;

  // -------------------------------------------------------------------------
  // Configured trainers
  std::vector< trainer_config > m_trainers;

  // Cached training data
  kv::category_hierarchy_sptr m_labels;
  std::vector< std::string > m_train_image_names;
  std::vector< kv::detected_object_set_sptr > m_train_groundtruth;
  std::vector< std::string > m_test_image_names;
  std::vector< kv::detected_object_set_sptr > m_test_groundtruth;

  // Memory-based data cache
  std::vector< kv::image_container_sptr > m_train_images;
  std::vector< kv::image_container_sptr > m_test_images;
  bool m_data_from_memory = false;

  // Computed statistics
  training_data_statistics m_stats;

  // Logging
  kv::logger_handle_t m_logger;

  // -------------------------------------------------------------------------
  // Helper methods
  void compute_statistics_from_groundtruth(
    const std::vector< kv::detected_object_set_sptr >& train_gt,
    const std::vector< kv::detected_object_set_sptr >& test_gt );

  double compute_iou( const kv::bounding_box_d& a, const kv::bounding_box_d& b ) const;

  bool check_hard_requirements( const trainer_config& tc ) const;

  double compute_preference_score( const trainer_config& tc ) const;

  std::vector< trainer_config* > select_trainers();

  void write_statistics_file() const;
};


double
adaptive_detector_trainer::priv::compute_iou(
  const kv::bounding_box_d& a, const kv::bounding_box_d& b ) const
{
  double x1 = std::max( a.min_x(), b.min_x() );
  double y1 = std::max( a.min_y(), b.min_y() );
  double x2 = std::min( a.max_x(), b.max_x() );
  double y2 = std::min( a.max_y(), b.max_y() );

  if( x2 <= x1 || y2 <= y1 )
  {
    return 0.0;
  }

  double intersection = ( x2 - x1 ) * ( y2 - y1 );
  double area_a = a.width() * a.height();
  double area_b = b.width() * b.height();
  double union_area = area_a + area_b - intersection;

  if( union_area <= 0 )
  {
    return 0.0;
  }

  return intersection / union_area;
}


void
adaptive_detector_trainer::priv::compute_statistics_from_groundtruth(
  const std::vector< kv::detected_object_set_sptr >& train_gt,
  const std::vector< kv::detected_object_set_sptr >& test_gt )
{
  m_stats = training_data_statistics();  // Reset

  // We'll estimate image dimensions from bounding boxes if not available
  // For edge detection, we'll use a heuristic based on max bbox coordinates

  double total_iou_sum = 0;
  size_t total_pairs = 0;
  size_t high_overlap_objects = 0;

  auto process_groundtruth = [&](
    const std::vector< kv::detected_object_set_sptr >& gt_list,
    bool is_train )
  {
    for( const auto& det_set : gt_list )
    {
      m_stats.total_frames++;

      if( !det_set || det_set->empty() )
      {
        m_stats.objects_per_frame.push_back( 0 );
        continue;
      }

      if( is_train )
        m_stats.train_frames_with_annotations++;
      else
        m_stats.test_frames_with_annotations++;

      size_t frame_object_count = det_set->size();
      m_stats.objects_per_frame.push_back( frame_object_count );

      // Collect all bboxes for this frame for overlap computation
      std::vector< kv::bounding_box_d > frame_boxes;

      // Estimate image bounds from this frame's detections
      double max_x = 0, max_y = 0;

      for( auto detection = det_set->cbegin();
           detection != det_set->cend();
           ++detection )
      {
        if( is_train )
          m_stats.total_train_annotations++;
        else
          m_stats.total_test_annotations++;

        const kv::bounding_box_d& bbox = (*detection)->bounding_box();
        double width = bbox.width();
        double height = bbox.height();
        double area = width * height;

        // Track max coordinates for edge detection
        if( bbox.max_x() > max_x ) max_x = bbox.max_x();
        if( bbox.max_y() > max_y ) max_y = bbox.max_y();

        m_stats.all_widths.push_back( width );
        m_stats.all_heights.push_back( height );
        m_stats.all_areas.push_back( area );

        // Aspect ratio
        if( height > 0 )
        {
          m_stats.all_aspect_ratios.push_back( width / height );
        }

        // Class label
        if( (*detection)->type() )
        {
          std::string label;
          (*detection)->type()->get_most_likely( label );
          if( !label.empty() )
          {
            m_stats.class_counts[ label ]++;
          }
        }

        // Check for mask
        if( (*detection)->mask() )
        {
          m_stats.objects_with_masks++;
        }

        frame_boxes.push_back( bbox );
      }

      // Edge detection using estimated image dimensions
      double edge_margin_x = max_x * m_edge_margin_fraction;
      double edge_margin_y = max_y * m_edge_margin_fraction;

      for( const auto& bbox : frame_boxes )
      {
        if( bbox.min_x() < edge_margin_x ||
            bbox.min_y() < edge_margin_y ||
            bbox.max_x() > ( max_x - edge_margin_x ) ||
            bbox.max_y() > ( max_y - edge_margin_y ) )
        {
          m_stats.edge_object_count++;
        }
      }

      // Compute pairwise IoU for overlap analysis
      for( size_t i = 0; i < frame_boxes.size(); ++i )
      {
        bool has_high_overlap = false;
        for( size_t j = i + 1; j < frame_boxes.size(); ++j )
        {
          double iou = compute_iou( frame_boxes[i], frame_boxes[j] );
          if( iou > 0 )
          {
            total_iou_sum += iou;
            total_pairs++;

            if( iou >= m_overlap_iou_threshold )
            {
              m_stats.overlapping_pair_count++;
              has_high_overlap = true;
            }
          }
        }
        if( has_high_overlap )
        {
          high_overlap_objects++;
        }
      }
    }
  };

  // Process both train and test sets
  process_groundtruth( train_gt, true );
  process_groundtruth( test_gt, false );

  // Compute mean IoU and overlap fraction
  if( total_pairs > 0 )
  {
    m_stats.mean_inter_object_iou = total_iou_sum / total_pairs;
  }

  size_t total_objects = m_stats.all_areas.size();
  if( total_objects > 0 )
  {
    m_stats.high_overlap_fraction =
      static_cast< double >( high_overlap_objects ) / total_objects;
  }

  // Compute summary statistics
  m_stats.compute_summary(
    m_small_object_threshold, m_large_object_threshold,
    m_tall_aspect_threshold, m_wide_aspect_threshold,
    m_crowded_frame_threshold, m_sparse_frame_threshold,
    m_rare_class_threshold, m_dominant_class_threshold,
    m_edge_margin_fraction, m_overlap_iou_threshold );

  if( m_verbose )
  {
    m_stats.log_statistics( m_logger );
  }
}


bool
adaptive_detector_trainer::priv::check_hard_requirements( const trainer_config& tc ) const
{
  // Check minimum count per class
  if( tc.required_min_count_per_class > 0 )
  {
    for( const auto& kv : m_stats.class_counts )
    {
      if( kv.second < tc.required_min_count_per_class )
      {
        if( m_verbose )
        {
          LOG_DEBUG( m_logger, "Trainer " << tc.name << " failed: class '"
                     << kv.first << "' has " << kv.second
                     << " annotations, requires " << tc.required_min_count_per_class );
        }
        return false;
      }
    }
  }

  // Check object size with percentile
  if( tc.required_min_object_area > 0 && !m_stats.all_areas.empty() )
  {
    size_t meeting_criteria = 0;
    for( double area : m_stats.all_areas )
    {
      if( area >= tc.required_min_object_area )
      {
        meeting_criteria++;
      }
    }

    double fraction = static_cast< double >( meeting_criteria ) / m_stats.all_areas.size();
    if( fraction < tc.required_percentile )
    {
      if( m_verbose )
      {
        LOG_DEBUG( m_logger, "Trainer " << tc.name << " failed: only "
                   << ( fraction * 100 ) << "% of objects meet min area "
                   << tc.required_min_object_area << ", requires "
                   << ( tc.required_percentile * 100 ) << "%" );
      }
      return false;
    }
  }

  // Check aspect ratio variance
  if( tc.required_max_aspect_ratio_std > 0 &&
      m_stats.aspect_ratio_std > tc.required_max_aspect_ratio_std )
  {
    if( m_verbose )
    {
      LOG_DEBUG( m_logger, "Trainer " << tc.name << " failed: aspect ratio std "
                 << m_stats.aspect_ratio_std << " exceeds max "
                 << tc.required_max_aspect_ratio_std );
    }
    return false;
  }

  // Check max objects per frame
  if( tc.required_max_objects_per_frame > 0 &&
      m_stats.max_objects_per_frame > tc.required_max_objects_per_frame )
  {
    if( m_verbose )
    {
      LOG_DEBUG( m_logger, "Trainer " << tc.name << " failed: max objects/frame "
                 << m_stats.max_objects_per_frame << " exceeds limit "
                 << tc.required_max_objects_per_frame );
    }
    return false;
  }

  // Check class imbalance
  if( tc.required_max_class_imbalance > 0 &&
      m_stats.class_imbalance_ratio > tc.required_max_class_imbalance )
  {
    if( m_verbose )
    {
      LOG_DEBUG( m_logger, "Trainer " << tc.name << " failed: class imbalance "
                 << m_stats.class_imbalance_ratio << " exceeds max "
                 << tc.required_max_class_imbalance );
    }
    return false;
  }

  // Check mask requirement
  if( tc.required_masks && !m_stats.has_masks )
  {
    if( m_verbose )
    {
      LOG_DEBUG( m_logger, "Trainer " << tc.name
                 << " failed: requires masks but data has insufficient masks ("
                 << ( m_stats.mask_fraction * 100 ) << "%)" );
    }
    return false;
  }

  return true;
}


double
adaptive_detector_trainer::priv::compute_preference_score( const trainer_config& tc ) const
{
  double score = 0.0;
  size_t total = m_stats.total_train_annotations + m_stats.total_test_annotations;
  size_t total_objects = m_stats.all_areas.size();

  // -------------------------------------------------------------------------
  // Annotation count preference
  if( !tc.annotation_count_preference.empty() )
  {
    if( tc.annotation_count_preference == "low" && total < m_low_annotation_threshold )
      score += 1.0;
    else if( tc.annotation_count_preference == "medium" &&
             total >= m_low_annotation_threshold && total < m_high_annotation_threshold )
      score += 1.0;
    else if( tc.annotation_count_preference == "high" && total >= m_high_annotation_threshold )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Object size preference
  if( !tc.object_size_preference.empty() && total_objects > 0 )
  {
    double small_frac = static_cast< double >( m_stats.small_object_count ) / total_objects;
    double medium_frac = static_cast< double >( m_stats.medium_object_count ) / total_objects;
    double large_frac = static_cast< double >( m_stats.large_object_count ) / total_objects;

    std::string dominant = "medium";
    double max_frac = medium_frac;
    if( small_frac > max_frac ) { dominant = "small"; max_frac = small_frac; }
    if( large_frac > max_frac ) { dominant = "large"; }

    if( tc.object_size_preference == dominant )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Aspect ratio preference
  if( !tc.aspect_ratio_preference.empty() && total_objects > 0 )
  {
    double tall_frac = static_cast< double >( m_stats.tall_object_count ) / total_objects;
    double square_frac = static_cast< double >( m_stats.square_object_count ) / total_objects;
    double wide_frac = static_cast< double >( m_stats.wide_object_count ) / total_objects;

    std::string dominant = "square";
    double max_frac = square_frac;
    if( tall_frac > max_frac ) { dominant = "tall"; max_frac = tall_frac; }
    if( wide_frac > max_frac ) { dominant = "wide"; }

    if( tc.aspect_ratio_preference == dominant )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Density preference
  if( !tc.density_preference.empty() )
  {
    bool is_sparse = ( m_stats.mean_objects_per_frame <= m_sparse_frame_threshold );
    bool is_dense = ( m_stats.mean_objects_per_frame >= m_crowded_frame_threshold );

    if( tc.density_preference == "sparse" && is_sparse )
      score += 1.0;
    else if( tc.density_preference == "dense" && is_dense )
      score += 1.0;
    else if( tc.density_preference == "medium" && !is_sparse && !is_dense )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Scale preference
  if( !tc.scale_preference.empty() )
  {
    if( tc.scale_preference == "multi-scale" && m_stats.is_multi_scale )
      score += 1.0;
    else if( tc.scale_preference == "uniform" && !m_stats.is_multi_scale )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Overlap preference
  if( !tc.overlap_preference.empty() )
  {
    bool low_overlap = ( m_stats.high_overlap_fraction < 0.1 );
    bool high_overlap = ( m_stats.high_overlap_fraction > 0.3 );

    if( tc.overlap_preference == "low" && low_overlap )
      score += 1.0;
    else if( tc.overlap_preference == "high" && high_overlap )
      score += 1.0;
    else if( tc.overlap_preference == "medium" && !low_overlap && !high_overlap )
      score += 1.0;
  }

  // -------------------------------------------------------------------------
  // Mask preference
  if( tc.prefers_masks && m_stats.has_masks )
  {
    score += 1.0;
  }

  return score;
}


std::vector< trainer_config* >
adaptive_detector_trainer::priv::select_trainers()
{
  std::vector< trainer_config* > qualifying;

  for( auto& tc : m_trainers )
  {
    if( !tc.trainer )
    {
      continue;
    }

    if( check_hard_requirements( tc ) )
    {
      tc.preference_score = compute_preference_score( tc );
      qualifying.push_back( &tc );

      if( m_verbose )
      {
        LOG_INFO( m_logger, "Trainer " << tc.name << " qualifies with preference score "
                  << tc.preference_score );
      }
    }
    else
    {
      if( m_verbose )
      {
        LOG_INFO( m_logger, "Trainer " << tc.name << " does not meet hard requirements" );
      }
    }
  }

  // Sort by preference score (descending)
  std::sort( qualifying.begin(), qualifying.end(),
    []( const trainer_config* a, const trainer_config* b )
    {
      return a->preference_score > b->preference_score;
    } );

  // Limit to max_trainers_to_run
  if( qualifying.size() > m_max_trainers_to_run )
  {
    qualifying.resize( m_max_trainers_to_run );
  }

  return qualifying;
}


void
adaptive_detector_trainer::priv::write_statistics_file() const
{
  std::ofstream out( m_output_statistics_file );
  if( !out.is_open() )
  {
    LOG_WARN( m_logger, "Could not open statistics file: " << m_output_statistics_file );
    return;
  }

  out << std::fixed << std::setprecision( 4 );

  out << "{\n";

  // Annotation counts
  out << "  \"annotation_counts\": {\n";
  out << "    \"total_train\": " << m_stats.total_train_annotations << ",\n";
  out << "    \"total_test\": " << m_stats.total_test_annotations << ",\n";
  out << "    \"total_frames\": " << m_stats.total_frames << ",\n";
  out << "    \"train_frames_with_annotations\": " << m_stats.train_frames_with_annotations << ",\n";
  out << "    \"test_frames_with_annotations\": " << m_stats.test_frames_with_annotations << "\n";
  out << "  },\n";

  // Class counts
  out << "  \"class_counts\": {\n";
  bool first = true;
  for( const auto& kv : m_stats.class_counts )
  {
    if( !first ) out << ",\n";
    out << "    \"" << kv.first << "\": " << kv.second;
    first = false;
  }
  out << "\n  },\n";

  // Object sizes
  out << "  \"object_sizes\": {\n";
  out << "    \"area\": { \"min\": " << m_stats.min_area << ", \"max\": " << m_stats.max_area
      << ", \"mean\": " << m_stats.mean_area << ", \"median\": " << m_stats.median_area << " },\n";
  out << "    \"percentiles\": { \"p10\": " << m_stats.area_10th_percentile
      << ", \"p50\": " << m_stats.area_50th_percentile
      << ", \"p90\": " << m_stats.area_90th_percentile << " },\n";
  out << "    \"distribution\": { \"small\": " << m_stats.small_object_count
      << ", \"medium\": " << m_stats.medium_object_count
      << ", \"large\": " << m_stats.large_object_count << " }\n";
  out << "  },\n";

  // Aspect ratios
  out << "  \"aspect_ratios\": {\n";
  out << "    \"mean\": " << m_stats.mean_aspect_ratio << ",\n";
  out << "    \"std\": " << m_stats.aspect_ratio_std << ",\n";
  out << "    \"distribution\": { \"tall\": " << m_stats.tall_object_count
      << ", \"square\": " << m_stats.square_object_count
      << ", \"wide\": " << m_stats.wide_object_count << " }\n";
  out << "  },\n";

  // Density
  out << "  \"density\": {\n";
  out << "    \"mean_objects_per_frame\": " << m_stats.mean_objects_per_frame << ",\n";
  out << "    \"max_objects_per_frame\": " << m_stats.max_objects_per_frame << ",\n";
  out << "    \"crowded_frames\": " << m_stats.crowded_frame_count << ",\n";
  out << "    \"sparse_frames\": " << m_stats.sparse_frame_count << "\n";
  out << "  },\n";

  // Scale
  out << "  \"scale\": {\n";
  out << "    \"variance\": " << m_stats.scale_variance << ",\n";
  out << "    \"min_max_ratio\": " << m_stats.min_max_area_ratio << ",\n";
  out << "    \"is_multi_scale\": " << ( m_stats.is_multi_scale ? "true" : "false" ) << "\n";
  out << "  },\n";

  // Class imbalance
  out << "  \"class_imbalance\": {\n";
  out << "    \"ratio\": " << m_stats.class_imbalance_ratio << ",\n";
  out << "    \"rare_classes\": " << m_stats.rare_class_count << ",\n";
  out << "    \"dominant_classes\": " << m_stats.dominant_class_count << "\n";
  out << "  },\n";

  // Spatial
  out << "  \"spatial\": {\n";
  out << "    \"edge_objects\": " << m_stats.edge_object_count << ",\n";
  out << "    \"edge_fraction\": " << m_stats.edge_object_fraction << "\n";
  out << "  },\n";

  // Overlap
  out << "  \"overlap\": {\n";
  out << "    \"mean_iou\": " << m_stats.mean_inter_object_iou << ",\n";
  out << "    \"overlapping_pairs\": " << m_stats.overlapping_pair_count << ",\n";
  out << "    \"high_overlap_fraction\": " << m_stats.high_overlap_fraction << "\n";
  out << "  },\n";

  // Masks
  out << "  \"masks\": {\n";
  out << "    \"objects_with_masks\": " << m_stats.objects_with_masks << ",\n";
  out << "    \"mask_fraction\": " << m_stats.mask_fraction << ",\n";
  out << "    \"has_masks\": " << ( m_stats.has_masks ? "true" : "false" ) << "\n";
  out << "  }\n";

  out << "}\n";

  out.close();
  LOG_INFO( m_logger, "Wrote statistics to: " << m_output_statistics_file );
}


// =============================================================================
adaptive_detector_trainer
::adaptive_detector_trainer()
  : d( new priv() )
{
  attach_logger( "viame.core.adaptive_detector_trainer" );
  d->m_logger = logger();
}


adaptive_detector_trainer
::~adaptive_detector_trainer()
{
}


// -----------------------------------------------------------------------------
kv::config_block_sptr
adaptive_detector_trainer
::get_configuration() const
{
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  // -------------------------------------------------------------------------
  // Global settings
  config->set_value( "max_trainers_to_run", d->m_max_trainers_to_run,
    "Maximum number of trainers to run sequentially. Default: 3" );

  // Size thresholds
  config->set_value( "small_object_threshold", d->m_small_object_threshold,
    "Area threshold (pixels^2) below which objects are 'small'. Default: 1024" );
  config->set_value( "large_object_threshold", d->m_large_object_threshold,
    "Area threshold (pixels^2) above which objects are 'large'. Default: 16384" );

  // Aspect ratio thresholds
  config->set_value( "tall_aspect_threshold", d->m_tall_aspect_threshold,
    "Aspect ratio (w/h) below which objects are 'tall'. Default: 0.5" );
  config->set_value( "wide_aspect_threshold", d->m_wide_aspect_threshold,
    "Aspect ratio (w/h) above which objects are 'wide'. Default: 2.0" );

  // Annotation count thresholds
  config->set_value( "low_annotation_threshold", d->m_low_annotation_threshold,
    "Annotation count below which datasets are 'low' data. Default: 500" );
  config->set_value( "high_annotation_threshold", d->m_high_annotation_threshold,
    "Annotation count above which datasets are 'high' data. Default: 2000" );

  // Density thresholds
  config->set_value( "sparse_frame_threshold", d->m_sparse_frame_threshold,
    "Max objects/frame for 'sparse' classification. Default: 5" );
  config->set_value( "crowded_frame_threshold", d->m_crowded_frame_threshold,
    "Min objects/frame for 'crowded' classification. Default: 20" );

  // Class thresholds
  config->set_value( "rare_class_threshold", d->m_rare_class_threshold,
    "Count below which a class is 'rare'. Default: 50" );
  config->set_value( "dominant_class_threshold", d->m_dominant_class_threshold,
    "Count above which a class is 'dominant'. Default: 500" );

  // Edge detection
  config->set_value( "edge_margin_fraction", d->m_edge_margin_fraction,
    "Fraction of image dimension for edge margin. Default: 0.05" );

  // Overlap threshold
  config->set_value( "overlap_iou_threshold", d->m_overlap_iou_threshold,
    "IoU threshold for 'high overlap' classification. Default: 0.3" );

  // Output
  config->set_value( "output_statistics_file", d->m_output_statistics_file,
    "Optional file path for JSON statistics. Empty = disabled." );
  config->set_value( "verbose", d->m_verbose,
    "Enable verbose logging. Default: true" );

  // -------------------------------------------------------------------------
  // Trainer configurations
  for( size_t i = 0; i < d->m_trainers.size(); ++i )
  {
    const auto& tc = d->m_trainers[i];
    std::string prefix = "trainer_" + std::to_string( i + 1 ) + ":";

    // Hard requirements
    config->set_value( prefix + "required_min_count_per_class", tc.required_min_count_per_class,
      "Minimum annotations per class. 0 = no requirement." );
    config->set_value( prefix + "required_min_object_area", tc.required_min_object_area,
      "Minimum object area. 0 = no requirement." );
    config->set_value( prefix + "required_percentile", tc.required_percentile,
      "Fraction of objects that must meet size criteria. Default: 0.5" );
    config->set_value( prefix + "required_max_aspect_ratio_std", tc.required_max_aspect_ratio_std,
      "Max allowed aspect ratio std dev. 0 = no requirement." );
    config->set_value( prefix + "required_max_objects_per_frame", tc.required_max_objects_per_frame,
      "Max allowed objects per frame. 0 = no requirement." );
    config->set_value( prefix + "required_max_class_imbalance", tc.required_max_class_imbalance,
      "Max allowed class imbalance ratio. 0 = no requirement." );
    config->set_value( prefix + "required_masks", tc.required_masks,
      "Require majority of objects to have masks. Default: false" );

    // Soft preferences
    config->set_value( prefix + "annotation_count_preference", tc.annotation_count_preference,
      "Preference: 'low', 'medium', 'high', or empty." );
    config->set_value( prefix + "object_size_preference", tc.object_size_preference,
      "Preference: 'small', 'medium', 'large', or empty." );
    config->set_value( prefix + "aspect_ratio_preference", tc.aspect_ratio_preference,
      "Preference: 'tall', 'square', 'wide', or empty." );
    config->set_value( prefix + "density_preference", tc.density_preference,
      "Preference: 'sparse', 'medium', 'dense', or empty." );
    config->set_value( prefix + "scale_preference", tc.scale_preference,
      "Preference: 'uniform', 'multi-scale', or empty." );
    config->set_value( prefix + "overlap_preference", tc.overlap_preference,
      "Preference: 'low', 'medium', 'high', or empty." );
    config->set_value( prefix + "prefers_masks", tc.prefers_masks,
      "Prefer datasets with masks. Default: false" );

    kv::algo::train_detector::get_nested_algo_configuration(
      prefix + "trainer", config, tc.trainer );
  }

  return config;
}


// -----------------------------------------------------------------------------
void
adaptive_detector_trainer
::set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  // -------------------------------------------------------------------------
  // Global settings
  d->m_max_trainers_to_run = config->get_value< size_t >( "max_trainers_to_run" );
  d->m_small_object_threshold = config->get_value< double >( "small_object_threshold" );
  d->m_large_object_threshold = config->get_value< double >( "large_object_threshold" );
  d->m_tall_aspect_threshold = config->get_value< double >( "tall_aspect_threshold" );
  d->m_wide_aspect_threshold = config->get_value< double >( "wide_aspect_threshold" );
  d->m_low_annotation_threshold = config->get_value< size_t >( "low_annotation_threshold" );
  d->m_high_annotation_threshold = config->get_value< size_t >( "high_annotation_threshold" );
  d->m_sparse_frame_threshold = config->get_value< size_t >( "sparse_frame_threshold" );
  d->m_crowded_frame_threshold = config->get_value< size_t >( "crowded_frame_threshold" );
  d->m_rare_class_threshold = config->get_value< size_t >( "rare_class_threshold" );
  d->m_dominant_class_threshold = config->get_value< size_t >( "dominant_class_threshold" );
  d->m_edge_margin_fraction = config->get_value< double >( "edge_margin_fraction" );
  d->m_overlap_iou_threshold = config->get_value< double >( "overlap_iou_threshold" );
  d->m_output_statistics_file = config->get_value< std::string >( "output_statistics_file" );
  d->m_verbose = config->get_value< bool >( "verbose" );

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

    trainer_config tc;
    tc.name = "trainer_" + std::to_string( i );

    // Hard requirements
    tc.required_min_count_per_class =
      config->get_value< size_t >( prefix + "required_min_count_per_class", 0 );
    tc.required_min_object_area =
      config->get_value< double >( prefix + "required_min_object_area", 0.0 );
    tc.required_percentile =
      config->get_value< double >( prefix + "required_percentile", 0.5 );
    tc.required_max_aspect_ratio_std =
      config->get_value< double >( prefix + "required_max_aspect_ratio_std", 0.0 );
    tc.required_max_objects_per_frame =
      config->get_value< double >( prefix + "required_max_objects_per_frame", 0.0 );
    tc.required_max_class_imbalance =
      config->get_value< double >( prefix + "required_max_class_imbalance", 0.0 );
    tc.required_masks =
      config->get_value< bool >( prefix + "required_masks", false );

    // Soft preferences
    tc.annotation_count_preference =
      config->get_value< std::string >( prefix + "annotation_count_preference", "" );
    tc.object_size_preference =
      config->get_value< std::string >( prefix + "object_size_preference", "" );
    tc.aspect_ratio_preference =
      config->get_value< std::string >( prefix + "aspect_ratio_preference", "" );
    tc.density_preference =
      config->get_value< std::string >( prefix + "density_preference", "" );
    tc.scale_preference =
      config->get_value< std::string >( prefix + "scale_preference", "" );
    tc.overlap_preference =
      config->get_value< std::string >( prefix + "overlap_preference", "" );
    tc.prefers_masks =
      config->get_value< bool >( prefix + "prefers_masks", false );

    // Nested trainer
    kv::algo::train_detector_sptr trainer;
    kv::algo::train_detector::set_nested_algo_configuration( trainer_key, config, trainer );
    tc.trainer = trainer;

    if( tc.trainer )
    {
      d->m_trainers.push_back( tc );
      LOG_DEBUG( d->m_logger, "Loaded trainer configuration: " << tc.name );
    }
    else
    {
      LOG_WARN( d->m_logger, "Failed to create trainer: " << tc.name );
    }
  }

  LOG_INFO( d->m_logger, "Configured " << d->m_trainers.size() << " trainers" );
}


// -----------------------------------------------------------------------------
bool
adaptive_detector_trainer
::check_configuration( kv::config_block_sptr config ) const
{
  for( size_t i = 1; i <= 100; ++i )
  {
    std::string trainer_key = "trainer_" + std::to_string( i ) + ":trainer";
    if( config->has_value( trainer_key + ":type" ) )
    {
      if( kv::algo::train_detector::check_nested_algo_configuration( trainer_key, config ) )
      {
        return true;
      }
    }
    else
    {
      break;
    }
  }

  LOG_ERROR( logger(), "No valid trainers configured." );
  return false;
}


// -----------------------------------------------------------------------------
void
adaptive_detector_trainer
::add_data_from_disk(
  kv::category_hierarchy_sptr object_labels,
  std::vector< std::string > train_image_names,
  std::vector< kv::detected_object_set_sptr > train_groundtruth,
  std::vector< std::string > test_image_names,
  std::vector< kv::detected_object_set_sptr > test_groundtruth )
{
  d->m_labels = object_labels;
  d->m_train_image_names = train_image_names;
  d->m_train_groundtruth = train_groundtruth;
  d->m_test_image_names = test_image_names;
  d->m_test_groundtruth = test_groundtruth;
  d->m_data_from_memory = false;

  LOG_INFO( d->m_logger, "Analyzing training data statistics..." );
  d->compute_statistics_from_groundtruth( train_groundtruth, test_groundtruth );

  if( !d->m_output_statistics_file.empty() )
  {
    d->write_statistics_file();
  }
}


// -----------------------------------------------------------------------------
void
adaptive_detector_trainer
::add_data_from_memory(
  kv::category_hierarchy_sptr object_labels,
  std::vector< kv::image_container_sptr > train_images,
  std::vector< kv::detected_object_set_sptr > train_groundtruth,
  std::vector< kv::image_container_sptr > test_images,
  std::vector< kv::detected_object_set_sptr > test_groundtruth )
{
  d->m_labels = object_labels;
  d->m_train_images = train_images;
  d->m_train_groundtruth = train_groundtruth;
  d->m_test_images = test_images;
  d->m_test_groundtruth = test_groundtruth;
  d->m_data_from_memory = true;

  LOG_INFO( d->m_logger, "Analyzing training data statistics..." );
  d->compute_statistics_from_groundtruth( train_groundtruth, test_groundtruth );

  if( !d->m_output_statistics_file.empty() )
  {
    d->write_statistics_file();
  }
}


// -----------------------------------------------------------------------------
std::map<std::string, std::string>
adaptive_detector_trainer
::update_model()
{
  std::map<std::string, std::string> combined_output;

  if( d->m_trainers.empty() )
  {
    throw std::runtime_error( "No trainers configured." );
  }

  std::vector< trainer_config* > selected = d->select_trainers();

  if( selected.empty() )
  {
    throw std::runtime_error( "No trainers qualify based on current data." );
  }

  LOG_INFO( d->m_logger, "Running " << selected.size() << " trainer(s) sequentially" );

  for( size_t i = 0; i < selected.size(); ++i )
  {
    trainer_config* tc = selected[i];
    LOG_INFO( d->m_logger, "=== Running trainer " << ( i + 1 ) << "/" << selected.size()
              << ": " << tc->name << " (score: " << tc->preference_score << ") ===" );

    if( d->m_data_from_memory )
    {
      tc->trainer->add_data_from_memory(
        d->m_labels,
        d->m_train_images, d->m_train_groundtruth,
        d->m_test_images, d->m_test_groundtruth );
    }
    else
    {
      tc->trainer->add_data_from_disk(
        d->m_labels,
        d->m_train_image_names, d->m_train_groundtruth,
        d->m_test_image_names, d->m_test_groundtruth );
    }

    std::map<std::string, std::string> trainer_output = tc->trainer->update_model();

    // Merge output from this trainer into combined output
    for( const auto& pair : trainer_output )
    {
      combined_output[pair.first] = pair.second;
    }

    LOG_INFO( d->m_logger, "Completed training for: " << tc->name );
  }

  LOG_INFO( d->m_logger, "All selected trainers completed." );

  return combined_output;
}

} // end namespace viame
