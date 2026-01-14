/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "adaptive_trainer.h"

#include <vital/util/cpu_timer.h>
#include <vital/types/image_container.h>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace viame {

namespace kv = kwiver::vital;

// =============================================================================
// Statistics structure for training data analysis
struct training_data_statistics
{
  // Annotation counts
  size_t total_train_annotations = 0;
  size_t total_test_annotations = 0;
  size_t train_frames_with_annotations = 0;
  size_t test_frames_with_annotations = 0;

  // Per-class annotation counts
  std::map< std::string, size_t > class_counts;

  // Object size data (raw values for percentile computation)
  std::vector< double > all_widths;
  std::vector< double > all_heights;
  std::vector< double > all_areas;

  // Computed summary statistics
  double min_width = 0, max_width = 0, mean_width = 0, median_width = 0;
  double min_height = 0, max_height = 0, mean_height = 0, median_height = 0;
  double min_area = 0, max_area = 0, mean_area = 0, median_area = 0;

  // Size distribution counts
  size_t small_object_count = 0;
  size_t medium_object_count = 0;
  size_t large_object_count = 0;

  // Percentiles
  double area_10th_percentile = 0;
  double area_50th_percentile = 0;
  double area_90th_percentile = 0;

  void compute_summary( double small_thresh, double large_thresh );
  void log_statistics( kv::logger_handle_t logger ) const;
};


void
training_data_statistics::compute_summary( double small_thresh, double large_thresh )
{
  if( all_areas.empty() )
  {
    return;
  }

  // Create sorted copies for percentile computation
  std::vector< double > sorted_widths = all_widths;
  std::vector< double > sorted_heights = all_heights;
  std::vector< double > sorted_areas = all_areas;

  std::sort( sorted_widths.begin(), sorted_widths.end() );
  std::sort( sorted_heights.begin(), sorted_heights.end() );
  std::sort( sorted_areas.begin(), sorted_areas.end() );

  size_t n = sorted_areas.size();

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

  // Percentiles
  area_10th_percentile = sorted_areas[ static_cast< size_t >( n * 0.1 ) ];
  area_50th_percentile = sorted_areas[ n / 2 ];
  area_90th_percentile = sorted_areas[ std::min( static_cast< size_t >( n * 0.9 ), n - 1 ) ];

  // Size distribution
  for( double area : all_areas )
  {
    if( area < small_thresh )
    {
      small_object_count++;
    }
    else if( area >= large_thresh )
    {
      large_object_count++;
    }
    else
    {
      medium_object_count++;
    }
  }
}


void
training_data_statistics::log_statistics( kv::logger_handle_t logger ) const
{
  size_t total = total_train_annotations + total_test_annotations;

  LOG_INFO( logger, "=== Training Data Statistics ===" );
  LOG_INFO( logger, "Total annotations: " << total
            << " (train: " << total_train_annotations
            << ", test: " << total_test_annotations << ")" );
  LOG_INFO( logger, "Frames with annotations: train=" << train_frames_with_annotations
            << ", test=" << test_frames_with_annotations );

  LOG_INFO( logger, "Per-class counts:" );
  for( const auto& kv : class_counts )
  {
    LOG_INFO( logger, "  " << kv.first << ": " << kv.second );
  }

  if( !all_areas.empty() )
  {
    LOG_INFO( logger, "Object sizes (area in pixels^2):" );
    LOG_INFO( logger, "  Min: " << min_area << ", Max: " << max_area
              << ", Mean: " << mean_area << ", Median: " << median_area );
    LOG_INFO( logger, "  10th percentile: " << area_10th_percentile
              << ", 90th percentile: " << area_90th_percentile );
    LOG_INFO( logger, "Size distribution:" );
    LOG_INFO( logger, "  Small: " << small_object_count
              << ", Medium: " << medium_object_count
              << ", Large: " << large_object_count );
  }
}


// =============================================================================
// Per-trainer configuration
struct trainer_config
{
  std::string name;
  kv::algo::train_detector_sptr trainer;

  // Hard requirements (must be met to run)
  size_t required_min_count_per_class = 0;  // 0 = no requirement
  double required_min_object_area = 0;       // 0 = no requirement
  double required_percentile = 0.5;          // Fraction of objects meeting size criteria

  // Soft preferences (for ranking)
  std::string annotation_count_preference;   // "low", "medium", "high", or empty
  std::string object_size_preference;        // "small", "medium", "large", or empty

  // Computed preference score
  double preference_score = 0.0;
};


// =============================================================================
class adaptive_trainer::priv
{
public:
  priv()
    : m_max_trainers_to_run( 3 )
    , m_small_object_threshold( 1024.0 )     // 32x32 pixels
    , m_large_object_threshold( 16384.0 )    // 128x128 pixels
    , m_low_annotation_threshold( 500 )
    , m_high_annotation_threshold( 2000 )
    , m_output_statistics_file( "" )
    , m_verbose( true )
  {}

  ~priv() {}

  // Global configuration
  size_t m_max_trainers_to_run;
  double m_small_object_threshold;
  double m_large_object_threshold;
  size_t m_low_annotation_threshold;
  size_t m_high_annotation_threshold;
  std::string m_output_statistics_file;
  bool m_verbose;

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

  // Helper methods
  void compute_statistics_from_groundtruth(
    const std::vector< kv::detected_object_set_sptr >& train_gt,
    const std::vector< kv::detected_object_set_sptr >& test_gt );

  bool check_hard_requirements( const trainer_config& tc ) const;

  double compute_preference_score( const trainer_config& tc ) const;

  std::vector< trainer_config* > select_trainers();

  void write_statistics_file() const;
};


void
adaptive_trainer::priv::compute_statistics_from_groundtruth(
  const std::vector< kv::detected_object_set_sptr >& train_gt,
  const std::vector< kv::detected_object_set_sptr >& test_gt )
{
  m_stats = training_data_statistics();  // Reset

  // Process training groundtruth
  for( const auto& det_set : train_gt )
  {
    if( !det_set || det_set->empty() )
    {
      continue;
    }

    m_stats.train_frames_with_annotations++;

    for( auto detection = det_set->cbegin();
         detection != det_set->cend();
         ++detection )
    {
      m_stats.total_train_annotations++;

      // Get bounding box dimensions
      const kv::bounding_box_d& bbox = (*detection)->bounding_box();
      double width = bbox.width();
      double height = bbox.height();
      double area = width * height;

      m_stats.all_widths.push_back( width );
      m_stats.all_heights.push_back( height );
      m_stats.all_areas.push_back( area );

      // Get class label for per-class statistics
      if( (*detection)->type() )
      {
        std::string label;
        (*detection)->type()->get_most_likely( label );
        if( !label.empty() )
        {
          m_stats.class_counts[ label ]++;
        }
      }
    }
  }

  // Process test groundtruth
  for( const auto& det_set : test_gt )
  {
    if( !det_set || det_set->empty() )
    {
      continue;
    }

    m_stats.test_frames_with_annotations++;

    for( auto detection = det_set->cbegin();
         detection != det_set->cend();
         ++detection )
    {
      m_stats.total_test_annotations++;

      const kv::bounding_box_d& bbox = (*detection)->bounding_box();
      double width = bbox.width();
      double height = bbox.height();
      double area = width * height;

      m_stats.all_widths.push_back( width );
      m_stats.all_heights.push_back( height );
      m_stats.all_areas.push_back( area );

      // Also count test annotations per class
      if( (*detection)->type() )
      {
        std::string label;
        (*detection)->type()->get_most_likely( label );
        if( !label.empty() )
        {
          m_stats.class_counts[ label ]++;
        }
      }
    }
  }

  // Compute summary statistics
  m_stats.compute_summary( m_small_object_threshold, m_large_object_threshold );

  if( m_verbose )
  {
    m_stats.log_statistics( m_logger );
  }
}


bool
adaptive_trainer::priv::check_hard_requirements( const trainer_config& tc ) const
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

  return true;
}


double
adaptive_trainer::priv::compute_preference_score( const trainer_config& tc ) const
{
  double score = 0.0;
  size_t total = m_stats.total_train_annotations + m_stats.total_test_annotations;

  // Annotation count preference
  if( !tc.annotation_count_preference.empty() )
  {
    if( tc.annotation_count_preference == "low" && total < m_low_annotation_threshold )
    {
      score += 1.0;
    }
    else if( tc.annotation_count_preference == "medium" &&
             total >= m_low_annotation_threshold &&
             total < m_high_annotation_threshold )
    {
      score += 1.0;
    }
    else if( tc.annotation_count_preference == "high" &&
             total >= m_high_annotation_threshold )
    {
      score += 1.0;
    }
  }

  // Object size preference (based on dominant size category)
  if( !tc.object_size_preference.empty() && !m_stats.all_areas.empty() )
  {
    size_t total_objects = m_stats.all_areas.size();
    double small_frac = static_cast< double >( m_stats.small_object_count ) / total_objects;
    double medium_frac = static_cast< double >( m_stats.medium_object_count ) / total_objects;
    double large_frac = static_cast< double >( m_stats.large_object_count ) / total_objects;

    // Find dominant category
    std::string dominant = "medium";
    double max_frac = medium_frac;
    if( small_frac > max_frac ) { dominant = "small"; max_frac = small_frac; }
    if( large_frac > max_frac ) { dominant = "large"; }

    if( tc.object_size_preference == dominant )
    {
      score += 1.0;
    }
  }

  return score;
}


std::vector< trainer_config* >
adaptive_trainer::priv::select_trainers()
{
  std::vector< trainer_config* > qualifying;

  // Check each trainer against hard requirements
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
adaptive_trainer::priv::write_statistics_file() const
{
  std::ofstream out( m_output_statistics_file );
  if( !out.is_open() )
  {
    LOG_WARN( m_logger, "Could not open statistics file: " << m_output_statistics_file );
    return;
  }

  // Write JSON format for easy parsing
  out << "{\n";
  out << "  \"annotation_counts\": {\n";
  out << "    \"total_train\": " << m_stats.total_train_annotations << ",\n";
  out << "    \"total_test\": " << m_stats.total_test_annotations << ",\n";
  out << "    \"train_frames_with_annotations\": "
      << m_stats.train_frames_with_annotations << ",\n";
  out << "    \"test_frames_with_annotations\": "
      << m_stats.test_frames_with_annotations << "\n";
  out << "  },\n";

  out << "  \"class_counts\": {\n";
  bool first = true;
  for( const auto& kv : m_stats.class_counts )
  {
    if( !first ) out << ",\n";
    out << "    \"" << kv.first << "\": " << kv.second;
    first = false;
  }
  out << "\n  },\n";

  out << "  \"object_sizes\": {\n";
  out << "    \"width\": { \"min\": " << m_stats.min_width
      << ", \"max\": " << m_stats.max_width
      << ", \"mean\": " << m_stats.mean_width
      << ", \"median\": " << m_stats.median_width << " },\n";
  out << "    \"height\": { \"min\": " << m_stats.min_height
      << ", \"max\": " << m_stats.max_height
      << ", \"mean\": " << m_stats.mean_height
      << ", \"median\": " << m_stats.median_height << " },\n";
  out << "    \"area\": { \"min\": " << m_stats.min_area
      << ", \"max\": " << m_stats.max_area
      << ", \"mean\": " << m_stats.mean_area
      << ", \"median\": " << m_stats.median_area << " }\n";
  out << "  },\n";

  out << "  \"size_distribution\": {\n";
  out << "    \"small_count\": " << m_stats.small_object_count << ",\n";
  out << "    \"medium_count\": " << m_stats.medium_object_count << ",\n";
  out << "    \"large_count\": " << m_stats.large_object_count << ",\n";
  out << "    \"area_percentiles\": {\n";
  out << "      \"p10\": " << m_stats.area_10th_percentile << ",\n";
  out << "      \"p50\": " << m_stats.area_50th_percentile << ",\n";
  out << "      \"p90\": " << m_stats.area_90th_percentile << "\n";
  out << "    }\n";
  out << "  }\n";
  out << "}\n";

  out.close();
  LOG_INFO( m_logger, "Wrote statistics to: " << m_output_statistics_file );
}


// =============================================================================
adaptive_trainer
::adaptive_trainer()
  : d( new priv() )
{
  attach_logger( "viame.core.adaptive_trainer" );
  d->m_logger = logger();
}


adaptive_trainer
::~adaptive_trainer()
{
}


// -----------------------------------------------------------------------------
kv::config_block_sptr
adaptive_trainer
::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  // Global settings
  config->set_value( "max_trainers_to_run", d->m_max_trainers_to_run,
    "Maximum number of trainers to run sequentially. Default: 3" );

  config->set_value( "small_object_threshold", d->m_small_object_threshold,
    "Area threshold (pixels^2) below which objects are considered 'small'. "
    "Default: 1024 (32x32 pixels)" );

  config->set_value( "large_object_threshold", d->m_large_object_threshold,
    "Area threshold (pixels^2) above which objects are considered 'large'. "
    "Default: 16384 (128x128 pixels)" );

  config->set_value( "low_annotation_threshold", d->m_low_annotation_threshold,
    "Annotation count below which datasets are considered 'low' data. "
    "Default: 500" );

  config->set_value( "high_annotation_threshold", d->m_high_annotation_threshold,
    "Annotation count above which datasets are considered 'high' data. "
    "Default: 2000" );

  config->set_value( "output_statistics_file", d->m_output_statistics_file,
    "Optional file path to write computed statistics in JSON format. "
    "Empty string disables output. Default: empty" );

  config->set_value( "verbose", d->m_verbose,
    "Enable verbose logging of statistics and selection decisions. "
    "Default: true" );

  // Export existing trainer configurations
  for( size_t i = 0; i < d->m_trainers.size(); ++i )
  {
    const auto& tc = d->m_trainers[i];
    std::string prefix = "trainer_" + std::to_string( i + 1 ) + ":";

    config->set_value( prefix + "required_min_count_per_class",
      tc.required_min_count_per_class,
      "Minimum annotations per class required to run this trainer. 0 = no requirement." );

    config->set_value( prefix + "required_min_object_area",
      tc.required_min_object_area,
      "Minimum object area (pixels^2) required. 0 = no requirement." );

    config->set_value( prefix + "required_percentile",
      tc.required_percentile,
      "Fraction of objects that must meet size criteria. Default: 0.5" );

    config->set_value( prefix + "annotation_count_preference",
      tc.annotation_count_preference,
      "Preference for annotation count: 'low', 'medium', 'high', or empty." );

    config->set_value( prefix + "object_size_preference",
      tc.object_size_preference,
      "Preference for object sizes: 'small', 'medium', 'large', or empty." );

    kv::algo::train_detector::get_nested_algo_configuration(
      prefix + "trainer", config, tc.trainer );
  }

  return config;
}


// -----------------------------------------------------------------------------
void
adaptive_trainer
::set_configuration( kv::config_block_sptr config_in )
{
  // Start with defaults
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  // Global settings
  d->m_max_trainers_to_run = config->get_value< size_t >( "max_trainers_to_run" );
  d->m_small_object_threshold = config->get_value< double >( "small_object_threshold" );
  d->m_large_object_threshold = config->get_value< double >( "large_object_threshold" );
  d->m_low_annotation_threshold = config->get_value< size_t >( "low_annotation_threshold" );
  d->m_high_annotation_threshold = config->get_value< size_t >( "high_annotation_threshold" );
  d->m_output_statistics_file = config->get_value< std::string >( "output_statistics_file" );
  d->m_verbose = config->get_value< bool >( "verbose" );

  // Clear existing trainers and scan for configured ones
  d->m_trainers.clear();

  // Look for trainer_1, trainer_2, etc.
  for( size_t i = 1; i <= 100; ++i )  // Reasonable upper limit
  {
    std::string prefix = "trainer_" + std::to_string( i ) + ":";
    std::string trainer_key = prefix + "trainer";

    // Check if this trainer is configured by looking for the type key
    if( !config->has_value( trainer_key + ":type" ) )
    {
      // No more trainers configured
      break;
    }

    trainer_config tc;
    tc.name = "trainer_" + std::to_string( i );

    // Read trainer-specific settings
    tc.required_min_count_per_class =
      config->get_value< size_t >( prefix + "required_min_count_per_class", 0 );
    tc.required_min_object_area =
      config->get_value< double >( prefix + "required_min_object_area", 0.0 );
    tc.required_percentile =
      config->get_value< double >( prefix + "required_percentile", 0.5 );
    tc.annotation_count_preference =
      config->get_value< std::string >( prefix + "annotation_count_preference", "" );
    tc.object_size_preference =
      config->get_value< std::string >( prefix + "object_size_preference", "" );

    // Create the nested trainer
    kv::algo::train_detector_sptr trainer;
    kv::algo::train_detector::set_nested_algo_configuration(
      trainer_key, config, trainer );
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
adaptive_trainer
::check_configuration( kv::config_block_sptr config ) const
{
  // Check that at least one trainer is configured
  for( size_t i = 1; i <= 100; ++i )
  {
    std::string trainer_key = "trainer_" + std::to_string( i ) + ":trainer";
    if( config->has_value( trainer_key + ":type" ) )
    {
      if( kv::algo::train_detector::check_nested_algo_configuration( trainer_key, config ) )
      {
        return true;  // At least one valid trainer
      }
    }
    else
    {
      break;  // No more trainers
    }
  }

  LOG_ERROR( logger(), "No valid trainers configured. Configure at least one "
             "trainer (e.g., trainer_1:trainer:type = ...)" );
  return false;
}


// -----------------------------------------------------------------------------
void
adaptive_trainer
::add_data_from_disk(
  kv::category_hierarchy_sptr object_labels,
  std::vector< std::string > train_image_names,
  std::vector< kv::detected_object_set_sptr > train_groundtruth,
  std::vector< std::string > test_image_names,
  std::vector< kv::detected_object_set_sptr > test_groundtruth )
{
  // Store data for later use
  d->m_labels = object_labels;
  d->m_train_image_names = train_image_names;
  d->m_train_groundtruth = train_groundtruth;
  d->m_test_image_names = test_image_names;
  d->m_test_groundtruth = test_groundtruth;
  d->m_data_from_memory = false;

  // Compute statistics from groundtruth
  LOG_INFO( d->m_logger, "Analyzing training data statistics..." );
  d->compute_statistics_from_groundtruth( train_groundtruth, test_groundtruth );

  // Write statistics if configured
  if( !d->m_output_statistics_file.empty() )
  {
    d->write_statistics_file();
  }
}


// -----------------------------------------------------------------------------
void
adaptive_trainer
::add_data_from_memory(
  kv::category_hierarchy_sptr object_labels,
  std::vector< kv::image_container_sptr > train_images,
  std::vector< kv::detected_object_set_sptr > train_groundtruth,
  std::vector< kv::image_container_sptr > test_images,
  std::vector< kv::detected_object_set_sptr > test_groundtruth )
{
  // Store data
  d->m_labels = object_labels;
  d->m_train_images = train_images;
  d->m_train_groundtruth = train_groundtruth;
  d->m_test_images = test_images;
  d->m_test_groundtruth = test_groundtruth;
  d->m_data_from_memory = true;

  // Compute statistics
  LOG_INFO( d->m_logger, "Analyzing training data statistics..." );
  d->compute_statistics_from_groundtruth( train_groundtruth, test_groundtruth );

  // Write statistics if configured
  if( !d->m_output_statistics_file.empty() )
  {
    d->write_statistics_file();
  }
}


// -----------------------------------------------------------------------------
void
adaptive_trainer
::update_model()
{
  if( d->m_trainers.empty() )
  {
    throw std::runtime_error( "No trainers configured. Configure at least one "
                              "trainer using trainer_1:trainer:type = ..." );
  }

  // Select trainers based on data statistics
  std::vector< trainer_config* > selected = d->select_trainers();

  if( selected.empty() )
  {
    throw std::runtime_error( "No trainers qualify based on current data. "
                              "Check hard requirements or lower thresholds." );
  }

  LOG_INFO( d->m_logger, "Running " << selected.size() << " trainer(s) sequentially" );

  // Run each selected trainer
  for( size_t i = 0; i < selected.size(); ++i )
  {
    trainer_config* tc = selected[i];
    LOG_INFO( d->m_logger, "=== Running trainer " << ( i + 1 ) << "/" << selected.size()
              << ": " << tc->name << " (preference score: " << tc->preference_score << ") ===" );

    // Pass data to trainer
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

    // Train the model
    tc->trainer->update_model();

    LOG_INFO( d->m_logger, "Completed training for: " << tc->name );
  }

  LOG_INFO( d->m_logger, "All selected trainers completed." );
}

} // end namespace viame
