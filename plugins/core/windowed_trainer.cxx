/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "windowed_trainer.h"
#include "windowed_utils.h"
#include "utilities_file.h"

#include <vital/util/cpu_timer.h>
#include <vital/algo/image_io.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/bounding_box.h>

#include <kwiversys/SystemTools.hxx>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <thread>
#include <mutex>
#include <random>
#include <algorithm>
#include <vector>

namespace viame {

namespace kv = kwiver::vital;

#ifdef WIN32
  const std::string div = "\\";
#else
  const std::string div = "/";
#endif

// =============================================================================
class windowed_trainer::priv
{
public:
  priv()
    : m_train_directory( "deep_training" )
    , m_chip_subdirectory( "cached_chips" )
    , m_chip_format( "png" )
    , m_skip_format( false )
    , m_chip_random_factor( -1.0 )
    , m_always_write_image( false )
    , m_ensure_standard( false )
    , m_overlap_required( 0.05 )
    , m_chips_w_gt_only( false )
    , m_max_neg_ratio( 0.0 )
    , m_random_validation( 0.0 )
    , m_ignore_category( "false_alarm" )
    , m_min_train_box_length( 0 )
    , m_min_train_box_edge_dist( 0 )
    , m_small_box_area( 0 )
    , m_small_action( "" )
    , m_chip_threads( 0 )
    , m_reuse_cache( false )
    , m_background_chip_ratio( 0.0 )
    , m_synthetic_labels( true )
    , m_detect_small( false )
  {
    // Set trainer-specific defaults (different from detector/refiner)
    m_settings.original_to_chip_size = true;
  }

  ~priv()
  {}

  // Common chip settings (shared with detector/refiner)
  window_settings m_settings;

  // Trainer-specific settings
  std::string m_train_directory;
  std::string m_chip_subdirectory;
  std::string m_chip_format;
  bool m_skip_format;
  double m_chip_random_factor;
  bool m_always_write_image;
  bool m_ensure_standard;
  double m_overlap_required;
  bool m_chips_w_gt_only;
  double m_max_neg_ratio;
  double m_random_validation;
  std::string m_ignore_category;
  int m_min_train_box_length;
  double m_min_train_box_edge_dist;
  int m_small_box_area;
  std::string m_small_action;

  // Parallel chip generation + on-disk chip caching
  int m_chip_threads;
  bool m_reuse_cache;

  // Random background chips kept, as a fraction of annotated chips per frame
  double m_background_chip_ratio;

  // Helper functions
  void format_images_from_disk(
    std::vector< std::string > image_names,
    std::vector< kv::detected_object_set_sptr > groundtruth,
    std::vector< std::string >& formatted_names,
    std::vector< kv::detected_object_set_sptr >& formatted_truth );

  void process_one_frame(
    unsigned fid,
    const std::vector< std::string >& image_names,
    const std::vector< kv::detected_object_set_sptr >& groundtruth,
    double negative_ds_factor,
    std::vector< std::string >& names,
    std::vector< kv::detected_object_set_sptr >& truth );

  void format_image_from_memory(
    const kv::image& image,
    kv::detected_object_set_sptr groundtruth,
    const rescale_option format_method,
    std::vector< std::string >& formatted_names,
    std::vector< kv::detected_object_set_sptr >& formatted_truth,
    const std::string& frame_tag,
    std::mt19937& rng );

  bool filter_detections_in_roi(
    kv::detected_object_set_sptr all_detections,
    kv::bounding_box_d region,
    kv::detected_object_set_sptr& filt_detections,
    bool* overlapped = nullptr );

  std::string generate_filename( const std::string& frame_tag, int chip_idx );

  void write_chip_to_disk( const std::string& filename, const kv::image& image );

  // Chip-cache (manifest) helpers
  std::string frame_tag_for( unsigned fid, const std::string& image_fn );
  std::string manifest_path( const std::string& frame_tag );
  bool load_manifest(
    const std::string& frame_tag,
    std::vector< std::string >& names,
    std::vector< kv::detected_object_set_sptr >& truth );
  void write_manifest(
    const std::string& frame_tag,
    const std::vector< std::string >& names,
    const std::vector< kv::detected_object_set_sptr >& truth );

  kv::category_hierarchy_sptr labels_without_ignored(
    kv::category_hierarchy_sptr in );

  std::mutex m_category_mutex;
  bool m_synthetic_labels;
  bool m_detect_small;
  kv::category_hierarchy_sptr m_labels;
  std::map< std::string, int > m_category_map;
  kv::algo::image_io_sptr m_image_io;
  kv::algo::train_detector_sptr m_trainer;
  kv::logger_handle_t m_logger;
};


// =============================================================================
windowed_trainer
::windowed_trainer()
  : d( new priv() )
{
  attach_logger( "viame.core.windowed_trainer" );

  d->m_logger = logger();
}

windowed_trainer
::~windowed_trainer()
{
}


// -----------------------------------------------------------------------------
kv::config_block_sptr
windowed_trainer
::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  // Trainer-specific settings
  config->set_value( "train_directory", d->m_train_directory,
    "Directory for all files used in training." );
  config->set_value( "chip_format", d->m_chip_format,
    "Image format for output chips." );
  config->set_value( "skip_format", d->m_skip_format,
    "Skip file formatting, assume that the train_directory is pre-populated "
    "with all files required for model training." );

  // Common chip settings (shared with detector/refiner)
  config->merge_config( d->m_settings.chip_config() );

  // Additional trainer-specific settings
  config->set_value( "chip_random_factor", d->m_chip_random_factor,
    "A percentage [0.0, 1.0] of chips to randomly use in training" );
  config->set_value( "always_write_image", d->m_always_write_image,
    "Always re-write images to training directory even if they already exist "
    "elsewhere on disk." );
  config->set_value( "ensure_standard", d->m_ensure_standard,
    "If images are not one of 3 common formats (jpg, jpeg, png) or 3 channel "
    "write them to the training directory even if they are elsewhere already" );
  config->set_value( "overlap_required", d->m_overlap_required,
    "Percentage of which a target must appear on a chip for it to be included "
    "as a training sample for said chip." );
  config->set_value( "chips_w_gt_only", d->m_chips_w_gt_only,
    "Only chips with valid groundtruth objects on them will be included in "
    "training." );
  config->set_value( "max_neg_ratio", d->m_max_neg_ratio,
    "Do not use more than this many more frames without groundtruth in "
    "training than there are frames with truth." );
  config->set_value( "random_validation", d->m_random_validation,
    "Randomly add this percentage of training frames to validation." );
  config->set_value( "ignore_category", d->m_ignore_category,
    "Ignore this category in training, but still include chips around it." );
  config->set_value( "min_train_box_length", d->m_min_train_box_length,
    "If a box resizes to smaller than this during training, the input frame "
    "will not be used in training." );
  config->set_value( "min_train_box_edge_dist", d->m_min_train_box_edge_dist,
    "If non-zero and a box is within a chip boundary adjusted by this many "
    "pixels, do not train on the chip." );
  config->set_value( "small_box_area", d->m_small_box_area,
    "If a box resizes to smaller than this during training, consider it a small "
    "detection which might lead to several modifications to it." );
  config->set_value( "small_action", d->m_small_action,
    "Action to take in the event that a detection is considered small. Can "
    "either be none, remove, or any other string which will over-ride the "
    "detection type to be that string." );
  config->set_value( "chip_threads", d->m_chip_threads,
    "Worker threads for chip generation (0 = auto, 1 = serial)." );
  config->set_value( "reuse_cache", d->m_reuse_cache,
    "Reuse existing chips/manifests in train_directory instead of regenerating "
    "(train_directory is not wiped at startup)." );
  config->set_value( "background_chip_ratio", d->m_background_chip_ratio,
    "With chips_w_gt_only, also keep this fraction of random background chips "
    "per frame, relative to annotated chips (0 = none)." );

  kv::algo::image_io::get_nested_algo_configuration( "image_reader",
    config, d->m_image_io );
  kv::algo::train_detector::get_nested_algo_configuration( "trainer",
    config, d->m_trainer );

  return config;
}


// -----------------------------------------------------------------------------
void
windowed_trainer
::set_configuration( kv::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  kv::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  // Trainer-specific settings
  d->m_train_directory = config->get_value< std::string >( "train_directory" );
  d->m_chip_format = config->get_value< std::string >( "chip_format" );
  d->m_skip_format = config->get_value< bool >( "skip_format" );

  // Common chip settings (shared with detector/refiner)
  d->m_settings.set_chip_config( config );

  // Additional trainer-specific settings
  d->m_chip_random_factor = config->get_value< double >( "chip_random_factor" );
  d->m_always_write_image = config->get_value< bool >( "always_write_image" );
  d->m_ensure_standard = config->get_value< bool >( "ensure_standard" );
  d->m_overlap_required = config->get_value< double >( "overlap_required" );
  d->m_chips_w_gt_only = config->get_value< bool >( "chips_w_gt_only" );
  d->m_max_neg_ratio = config->get_value< double >( "max_neg_ratio" );
  d->m_random_validation = config->get_value< double >( "random_validation" );
  d->m_ignore_category = config->get_value< std::string >( "ignore_category" );
  d->m_min_train_box_length = config->get_value< int >( "min_train_box_length" );
  d->m_min_train_box_edge_dist = config->get_value< double >( "min_train_box_edge_dist" );
  d->m_small_box_area = config->get_value< int >( "small_box_area" );
  d->m_small_action = config->get_value< std::string >( "small_action" );
  d->m_chip_threads = config->get_value< int >( "chip_threads" );
  d->m_reuse_cache = config->get_value< bool >( "reuse_cache" );
  d->m_background_chip_ratio = config->get_value< double >( "background_chip_ratio" );

  if( !d->m_skip_format )
  {
    // Delete and reset folder contents, unless reusing a prior chip cache
    if( !d->m_reuse_cache &&
        kwiversys::SystemTools::FileExists( d->m_train_directory ) &&
        kwiversys::SystemTools::FileIsDirectory( d->m_train_directory ) )
    {
      kwiversys::SystemTools::RemoveADirectory( d->m_train_directory );

#ifndef WIN32
      if( kwiversys::SystemTools::FileExists( d->m_train_directory ) )
      {
        LOG_ERROR( d->m_logger, "Unable to delete pre-existing training dir" );
        return;
      }
#endif
    }

    kwiversys::SystemTools::MakeDirectory( d->m_train_directory );

    if( !d->m_chip_subdirectory.empty() )
    {
      std::string folder = d->m_train_directory + div + d->m_chip_subdirectory;
      kwiversys::SystemTools::MakeDirectory( folder );
    }
  }

  d->m_detect_small = ( !d->m_small_action.empty() && d->m_small_action != "none" );

  kv::algo::image_io_sptr io;
  kv::algo::image_io::set_nested_algo_configuration( "image_reader", config, io );
  d->m_image_io = io;

  // Nested trainers keep their own train_directory, defaulting to
  // "deep_training" independently of ours. Left alone they silently write
  // their datasets and checkpoints under that default while we chip into the
  // configured directory, so overriding train_directory here would scatter one
  // run across two folders and clobber whatever lives in "deep_training".
  // Hand our value down unless the config names one explicitly.
  const std::string trainer_type =
    config->get_value< std::string >( "trainer:type", "" );

  if( !trainer_type.empty() )
  {
    const std::string trainer_dir_key =
      "trainer:" + trainer_type + ":train_directory";

    if( !config->has_value( trainer_dir_key ) )
    {
      config->set_value( trainer_dir_key, d->m_train_directory );
    }
  }

  kv::algo::train_detector_sptr trainer;
  kv::algo::train_detector::set_nested_algo_configuration( "trainer", config, trainer );
  d->m_trainer = trainer;
}


// -----------------------------------------------------------------------------
bool
windowed_trainer
::check_configuration( kv::config_block_sptr config ) const
{
  return kv::algo::image_io::check_nested_algo_configuration(
     "image_reader", config )
   && kv::algo::train_detector::check_nested_algo_configuration(
     "trainer", config );
}


// -----------------------------------------------------------------------------
void
windowed_trainer
::add_data_from_disk(
  kv::category_hierarchy_sptr object_labels,
  std::vector< std::string > train_image_names,
  std::vector< kv::detected_object_set_sptr > train_groundtruth,
  std::vector< std::string > test_image_names,
  std::vector< kv::detected_object_set_sptr > test_groundtruth)
{
  if( object_labels )
  {
    d->m_labels = object_labels;
    d->m_synthetic_labels = false;
  }

  std::vector< std::string > filtered_train_names;
  std::vector< kv::detected_object_set_sptr > filtered_train_truth;
  std::vector< std::string > filtered_test_names;
  std::vector< kv::detected_object_set_sptr > filtered_test_truth;

  if( !d->m_skip_format )
  {
    d->format_images_from_disk(
      train_image_names, train_groundtruth,
      filtered_train_names, filtered_train_truth );

    d->format_images_from_disk(
      test_image_names, test_groundtruth,
      filtered_test_names, filtered_test_truth );
  }

  if( d->m_synthetic_labels )
  {
    kv::category_hierarchy_sptr all_labels =
      std::make_shared< kv::category_hierarchy >();

    for( auto p = d->m_category_map.begin(); p != d->m_category_map.end(); p++ )
    {
      all_labels->add_class( p->first );
    }

    d->m_trainer->add_data_from_disk(
      all_labels,
      filtered_train_names, filtered_train_truth,
      filtered_test_names, filtered_test_truth );
  }
  else
  {
    d->m_trainer->add_data_from_disk(
      d->labels_without_ignored( object_labels ),
      filtered_train_names, filtered_train_truth,
      filtered_test_names, filtered_test_truth );
  }
}

void
windowed_trainer
::add_data_from_memory(
  kv::category_hierarchy_sptr object_labels,
  std::vector< kv::image_container_sptr > train_images,
  std::vector< kv::detected_object_set_sptr > train_groundtruth,
  std::vector< kv::image_container_sptr > test_images,
  std::vector< kv::detected_object_set_sptr > test_groundtruth)
{
  if( object_labels )
  {
    d->m_labels = object_labels;
    d->m_synthetic_labels = false;
  }

  std::vector< std::string > filtered_train_names;
  std::vector< kv::detected_object_set_sptr > filtered_train_truth;
  std::vector< std::string > filtered_test_names;
  std::vector< kv::detected_object_set_sptr > filtered_test_truth;

  if( !d->m_skip_format )
  {
    for( unsigned i = 0; i < train_images.size(); ++i )
    {
      kv::image image = train_images[i]->get_image();
      std::mt19937 rng( static_cast< uint64_t >( i ) * 2654435761ull + 1ull );

      if( d->m_random_validation > 0.0 &&
          static_cast< double >( rand() ) / RAND_MAX <= d->m_random_validation )
      {
        d->format_image_from_memory(
          image, train_groundtruth[i], d->m_settings.mode,
          filtered_test_names, filtered_test_truth,
          "mem_test_" + std::to_string( i ), rng );
      }
      else
      {
        d->format_image_from_memory(
          image, train_groundtruth[i], d->m_settings.mode,
          filtered_train_names, filtered_train_truth,
          "mem_train_" + std::to_string( i ), rng );
      }
    }
    for( unsigned i = 0; i < test_images.size(); ++i )
    {
      kv::image image = test_images[i]->get_image();
      std::mt19937 rng( static_cast< uint64_t >( i ) * 2654435761ull + 7ull );

      d->format_image_from_memory(
        image, test_groundtruth[i], d->m_settings.mode,
        filtered_test_names, filtered_test_truth,
        "mem_test2_" + std::to_string( i ), rng );
    }
  }

  d->m_trainer->add_data_from_disk(
    d->labels_without_ignored( object_labels ),
    filtered_train_names, filtered_train_truth,
    filtered_test_names, filtered_test_truth );
}

std::map<std::string, std::string>
windowed_trainer
::update_model()
{
  std::map<std::string, std::string> nested_output = d->m_trainer->update_model();

  const std::string algo = "windowed";
  const std::string nested_prefix = algo + ":detector:";

  std::map<std::string, std::string> output;

  // Re-key nested trainer output so config entries land under
  // the correct .pipe path (e.g. windowed:detector:netharn:deployed).
  // File copy entries (value is an existing file) keep their original
  // key since that key is the destination filename, not a config path.
  // Special keys like "eval_folder" are also passed through unchanged.
  for( const auto& pair : nested_output )
  {
    if( !pair.second.empty() && does_file_exist( pair.second ) )
    {
      output[ pair.first ] = pair.second;
    }
    else if( pair.first == "eval_folder" && !pair.second.empty() &&
             does_folder_exist( pair.second ) )
    {
      // Pass through eval_folder key unchanged for directory copies
      output[ pair.first ] = pair.second;
    }
    else
    {
      output[ nested_prefix + pair.first ] = pair.second;
    }
  }

  // Add windowed trainer's own config entries
  output["type"] = algo;
  output[algo + ":mode"] = rescale_option_converter().to_string( d->m_settings.mode );
  output[algo + ":scale"] = std::to_string( d->m_settings.scale );
  output[algo + ":chip_width"] = std::to_string( d->m_settings.chip_width );
  output[algo + ":chip_height"] = std::to_string( d->m_settings.chip_height );
  output[algo + ":chip_step_width"] = std::to_string( d->m_settings.chip_step_width );
  output[algo + ":chip_step_height"] = std::to_string( d->m_settings.chip_step_height );
  output[algo + ":chip_adaptive_thresh"] = std::to_string( d->m_settings.chip_adaptive_thresh );
  output[algo + ":original_to_chip_size"] = d->m_settings.original_to_chip_size ? "true" : "false";
  output[algo + ":black_pad"] = d->m_settings.black_pad ? "true" : "false";

  return output;
}

// -----------------------------------------------------------------------------
void
windowed_trainer::priv
::format_images_from_disk(
  std::vector< std::string > image_names,
  std::vector< kv::detected_object_set_sptr > groundtruth,
  std::vector< std::string >& formatted_names,
  std::vector< kv::detected_object_set_sptr >& formatted_truth )
{
  double negative_ds_factor = -1.0;

  if( m_max_neg_ratio > 0.0 && groundtruth.size() > 10 )
  {
    unsigned gt = 0, no_gt = 0;

    for( unsigned i = 0; i < groundtruth.size(); ++i )
    {
      if( groundtruth[i] && !groundtruth[i]->empty() )
      {
        gt++;
      }
      else
      {
        no_gt++;
      }
    }

    if( no_gt > 0 && gt > 0 )
    {
      double current_ratio = static_cast< double >( no_gt ) / gt;

      if( current_ratio > m_max_neg_ratio )
      {
        negative_ds_factor = m_max_neg_ratio / current_ratio;
      }
    }
  }

  const unsigned n = static_cast< unsigned >( image_names.size() );

  // Per-frame buffers, merged in frame order for thread independence
  std::vector< std::vector< std::string > > frame_names( n );
  std::vector< std::vector< kv::detected_object_set_sptr > > frame_truth( n );

  unsigned num_threads = ( m_chip_threads > 0 )
    ? static_cast< unsigned >( m_chip_threads )
    : std::thread::hardware_concurrency();

  if( num_threads == 0 )
  {
    num_threads = 1;
  }
  if( num_threads > n )
  {
    num_threads = ( n > 0 ? n : 1 );
  }

  auto worker = [&]( unsigned tid )
  {
    for( unsigned fid = tid; fid < n; fid += num_threads )
    {
      process_one_frame( fid, image_names, groundtruth, negative_ds_factor,
        frame_names[fid], frame_truth[fid] );
    }
  };

  if( num_threads <= 1 )
  {
    worker( 0 );
  }
  else
  {
    std::vector< std::thread > pool;
    pool.reserve( num_threads );
    for( unsigned t = 0; t < num_threads; ++t )
    {
      pool.emplace_back( worker, t );
    }
    for( auto& th : pool )
    {
      th.join();
    }
  }

  for( unsigned fid = 0; fid < n; ++fid )
  {
    for( unsigned k = 0; k < frame_names[fid].size(); ++k )
    {
      formatted_names.push_back( frame_names[fid][k] );
      formatted_truth.push_back( frame_truth[fid][k] );
    }
  }
}

void
windowed_trainer::priv
::process_one_frame(
  unsigned fid,
  const std::vector< std::string >& image_names,
  const std::vector< kv::detected_object_set_sptr >& groundtruth,
  double negative_ds_factor,
  std::vector< std::string >& names,
  std::vector< kv::detected_object_set_sptr >& truth )
{
  const std::string image_fn = image_names[fid];
  const std::string frame_tag = frame_tag_for( fid, image_fn );

  // Reuse cached frame if its manifest is present
  if( m_reuse_cache && load_manifest( frame_tag, names, truth ) )
  {
    return;
  }

  // Deterministic per-frame RNG: reproducible across runs, thread-safe
  std::mt19937 rng( static_cast< uint64_t >( fid ) * 2654435761ull + 11ull );
  std::uniform_real_distribution< double > unif( 0.0, 1.0 );

  if( negative_ds_factor > 0.0 &&
      ( !groundtruth[fid] || groundtruth[fid]->empty() ) &&
      unif( rng ) > negative_ds_factor )
  {
    return;
  }

  if( m_settings.mode == DISABLED && !m_always_write_image && !m_ensure_standard )
  {
    names.push_back( image_fn );
    truth.push_back( groundtruth[fid] );
    write_manifest( frame_tag, names, truth );
    return;
  }

  // Scale and break up image according to settings
  kv::image_container_sptr vital_image;
  kv::bounding_box_d image_dims;
  kv::image original_image;
  kv::detected_object_set_sptr filtered_truth;

  rescale_option format_mode = m_settings.mode;
  std::string ext = image_fn.substr( image_fn.find_last_of( "." ) + 1 );

  try
  {
    LOG_INFO( m_logger, "Loading image: " << image_fn );

    vital_image = m_image_io->load( image_fn );
    original_image = vital_image->get_image();

    image_dims = kv::bounding_box_d( 0, 0,
      original_image.width(), original_image.height() );
  }
  catch( const kv::vital_exception& e )
  {
    LOG_ERROR( m_logger, "Caught exception reading image: " << e.what() );
    return;
  }

  const int img_width = static_cast< int >( original_image.width() );
  const int img_height = static_cast< int >( original_image.height() );

  // Early exit don't need to read all images every iteration
  if( format_mode == ADAPTIVE )
  {
    if( ( img_height * img_width ) < m_settings.chip_adaptive_thresh )
    {
      if( m_always_write_image ||
          ( m_settings.original_to_chip_size &&
            ( img_width > m_settings.chip_width ||
              img_height > m_settings.chip_height ) ) ||
          ( m_ensure_standard &&
            ( original_image.depth() != 3 ||
             !( ext == "jpg" || ext == "png" || ext == "jpeg" ) ) ) )
      {
        format_mode = MAINTAIN_AR;
      }
      else
      {
        if( filter_detections_in_roi( groundtruth[fid], image_dims, filtered_truth ) )
        {
          names.push_back( image_fn );
          truth.push_back( filtered_truth );
        }
        write_manifest( frame_tag, names, truth );
        return;
      }
    }
    else
    {
      format_mode = CHIP_AND_ORIGINAL;
    }
  }
  else if( format_mode == ORIGINAL_AND_RESIZED )
  {
    if( img_height <= m_settings.chip_height && img_width <= m_settings.chip_width )
    {
      if( filter_detections_in_roi( groundtruth[fid], image_dims, filtered_truth ) )
      {
        names.push_back( image_fn );
        truth.push_back( filtered_truth );
      }
      write_manifest( frame_tag, names, truth );
      return;
    }

    format_mode = MAINTAIN_AR;

    if( ( img_height * img_width ) >= m_settings.chip_adaptive_thresh )
    {
      if( filter_detections_in_roi( groundtruth[fid], image_dims, filtered_truth ) )
      {
        names.push_back( image_fn );
        truth.push_back( filtered_truth );
      }
    }
  }

  // Format image and write new ones to disk
  format_image_from_memory(
    original_image, groundtruth[fid], format_mode,
    names, truth, frame_tag, rng );

  write_manifest( frame_tag, names, truth );
}

void
windowed_trainer::priv
::format_image_from_memory(
  const kv::image& image,
  kv::detected_object_set_sptr groundtruth,
  const rescale_option format_method,
  std::vector< std::string >& formatted_names,
  std::vector< kv::detected_object_set_sptr >& formatted_truth,
  const std::string& frame_tag,
  std::mt19937& rng )
{
  int chip_idx = 0;
  std::uniform_real_distribution< double > unif( 0.0, 1.0 );
  kv::image resized_image;
  kv::detected_object_set_sptr scaled_groundtruth = groundtruth->clone();
  kv::detected_object_set_sptr filtered_truth;

  double resized_scale = 1.0;

  if( format_method != DISABLED )
  {
    resized_image = format_image( image, format_method,
      m_settings.scale, m_settings.chip_width, m_settings.chip_height,
      m_settings.black_pad, resized_scale );

    scaled_groundtruth->scale( resized_scale );
  }
  else
  {
    resized_image = image;
    scaled_groundtruth = groundtruth;
  }

  const int resized_width = static_cast< int >( resized_image.width() );
  const int resized_height = static_cast< int >( resized_image.height() );

  if( format_method != CHIP && format_method != CHIP_AND_ORIGINAL )
  {
    kv::bounding_box_d roi_box( 0, 0, resized_width, resized_height );

    if( filter_detections_in_roi( scaled_groundtruth, roi_box, filtered_truth ) )
    {
      std::string img_file = generate_filename( frame_tag, chip_idx++ );
      write_chip_to_disk( img_file, resized_image );

      formatted_names.push_back( img_file );
      formatted_truth.push_back( filtered_truth );
    }
  }
  else
  {
    int annotated_chips = 0;
    std::vector< image_rect > background_rois;

    // Chip up and process scaled image
    for( int i = 0;
         i < resized_width - m_settings.chip_width + m_settings.chip_step_width;
         i += m_settings.chip_step_width )
    {
      int cw = i + m_settings.chip_width;

      if( cw > resized_width )
      {
        cw = resized_width - i;
      }
      else
      {
        cw = m_settings.chip_width;
      }

      for( int j = 0;
           j < resized_height - m_settings.chip_height + m_settings.chip_step_height;
           j += m_settings.chip_step_height )
      {
        // random downsampling
        if( m_chip_random_factor > 0.0 &&
              unif( rng ) > m_chip_random_factor )
        {
          continue;
        }

        int ch = j + m_settings.chip_height;

        if( ch > resized_height )
        {
          ch = resized_height - j;
        }
        else
        {
          ch = m_settings.chip_height;
        }

        // Only necessary in a few circumstances when chip_step exceeds image size.
        if( ch < 0 || cw < 0 )
        {
          continue;
        }

        image_rect roi( i, j, cw, ch );

        kv::bounding_box_d roi_box( i, j, i + m_settings.chip_width,
          j + m_settings.chip_height );

        bool overlapped = false;

        if( filter_detections_in_roi( scaled_groundtruth, roi_box,
              filtered_truth, &overlapped ) )
        {
          kv::image cropped_image = crop_image( resized_image, roi );

          double scaled_crop_scale;
          kv::image resized_crop = scale_image_maintaining_ar(
            cropped_image, m_settings.chip_width, m_settings.chip_height,
            m_settings.black_pad, scaled_crop_scale );

          std::string img_file = generate_filename( frame_tag, chip_idx++ );
          write_chip_to_disk( img_file, resized_crop );

          formatted_names.push_back( img_file );
          formatted_truth.push_back( filtered_truth );
          ++annotated_chips;
        }
        else if( m_background_chip_ratio > 0.0 && !overlapped )
        {
          background_rois.push_back( roi );
        }
      }
    }

    // Sample background chips proportional to annotated chips
    if( m_background_chip_ratio > 0.0 &&
        annotated_chips > 0 && !background_rois.empty() )
    {
      int target = static_cast< int >(
        m_background_chip_ratio * annotated_chips + 0.5 );
      target = std::min( target, static_cast< int >( background_rois.size() ) );

      std::shuffle( background_rois.begin(), background_rois.end(), rng );

      for( int n = 0; n < target; ++n )
      {
        const image_rect& roi = background_rois[n];
        kv::image cropped_image = crop_image( resized_image, roi );

        double scaled_crop_scale;
        kv::image resized_crop = scale_image_maintaining_ar(
          cropped_image, m_settings.chip_width, m_settings.chip_height,
          m_settings.black_pad, scaled_crop_scale );

        std::string img_file = generate_filename( frame_tag, chip_idx++ );
        write_chip_to_disk( img_file, resized_crop );

        formatted_names.push_back( img_file );
        formatted_truth.push_back( std::make_shared< kv::detected_object_set >() );
      }
    }

    // Process full sized image if enabled
    if( format_method == CHIP_AND_ORIGINAL )
    {
      double scaled_original_scale;
      kv::image scaled_original = scale_image_maintaining_ar( image,
        m_settings.chip_width, m_settings.chip_height, m_settings.black_pad,
        scaled_original_scale );

      kv::detected_object_set_sptr scaled_original_dets_ptr = groundtruth->clone();
      scaled_original_dets_ptr->scale( scaled_original_scale );

      kv::bounding_box_d roi_box( 0, 0, scaled_original.width(), scaled_original.height() );

      if( filter_detections_in_roi( scaled_original_dets_ptr, roi_box, filtered_truth ) )
      {
        std::string img_file = generate_filename( frame_tag, chip_idx++ );
        write_chip_to_disk( img_file, scaled_original );

        formatted_names.push_back( img_file );
        formatted_truth.push_back( filtered_truth );
      }
    }
  }
}


bool
windowed_trainer::priv
::filter_detections_in_roi(
  kv::detected_object_set_sptr all_detections,
  kv::bounding_box_d region,
  kv::detected_object_set_sptr& filtered_detections,
  bool* overlapped )
{
  auto ie = all_detections->cend();

  filtered_detections = std::make_shared< kv::detected_object_set >();

  // Did any annotation (incl. hard negatives) land on this chip
  bool had_overlap = false;

  if( overlapped )
  {
    *overlapped = false;
  }

  for( auto detection = all_detections->cbegin(); detection != ie; ++detection )
  {
    kv::bounding_box_d det_box = (*detection)->bounding_box();
    kv::bounding_box_d overlap = kv::intersection( region, det_box );

    if( det_box.width() < m_min_train_box_length ||
        det_box.height() < m_min_train_box_length )
    {
      return false;
    }

    if( det_box.area() > 0 &&
        overlap.max_x() > overlap.min_x() &&
        overlap.max_y() > overlap.min_y() &&
        overlap.area() / det_box.area() >= m_overlap_required )
    {
      had_overlap = true;

      if( overlapped )
      {
        *overlapped = true;
      }

      std::string category;

      if( !(*detection)->type() )
      {
        LOG_ERROR( m_logger, "Input detection is missing type category" );
        return false;
      }

      (*detection)->type()->get_most_likely( category );

      if( !m_ignore_category.empty() && category == m_ignore_category )
      {
        continue;
      }
      else if( m_synthetic_labels )
      {
        std::lock_guard< std::mutex > lock( m_category_mutex );
        if( m_category_map.find( category ) == m_category_map.end() )
        {
          m_category_map[ category ] = m_category_map.size() - 1;
        }
        category = std::to_string( m_category_map[ category ] );
      }
      else if( m_labels->has_class_name( category ) )
      {
        category = std::to_string( m_labels->get_class_id( category ) );
      }
      else
      {
        LOG_WARN( m_logger, "Ignoring unlisted class " << category );
        continue;
      }

      double min_x = det_box.min_x() - region.min_x();
      double min_y = det_box.min_y() - region.min_y();
      double max_x = det_box.max_x() - region.min_x();
      double max_y = det_box.max_y() - region.min_y();

      if( m_min_train_box_edge_dist != 0 &&
          ( min_x <= m_min_train_box_edge_dist ||
            min_y <= m_min_train_box_edge_dist ||
            max_x >= region.width() - m_min_train_box_edge_dist ||
            max_y >= region.height() - m_min_train_box_edge_dist ) )
      {
        return false;
      }

      kv::bounding_box_d bbox( min_x, min_y, max_x, max_y );

      auto odet = (*detection)->clone();
      odet->set_bounding_box( bbox );

      // Carry the segmentation polygon onto the chip: translate it into chip
      // coordinates (matching the bbox) and clamp to the chip bounds so the
      // rasterized mask stays inside the tile. clone() already copied the
      // (already-scaled) polygon, so we only translate + clamp here.
      auto poly = (*detection)->get_flattened_polygon();

      if( !poly.empty() )
      {
        for( size_t pi = 0; pi + 1 < poly.size(); pi += 2 )
        {
          double px = poly[pi] - region.min_x();
          double py = poly[pi + 1] - region.min_y();
          poly[pi] = std::max( 0.0, std::min( px, region.width() ) );
          poly[pi + 1] = std::max( 0.0, std::min( py, region.height() ) );
        }
        odet->set_flattened_polygon( poly );
      }

      if( m_detect_small && det_box.area() < m_small_box_area )
      {
        if( m_small_action == "remove" )
        {
          continue;
        }
        else if( m_small_action == "skip-chip" )
        {
          return false;
        }

        auto dot_ovr = std::make_shared< kv::detected_object_type >(
          m_small_action, 1.0 );

        odet->set_type( dot_ovr );
      }

      filtered_detections->add( odet );
    }
  }

  // Drop only chips with no annotation at all (hard-negative chips kept)
  if( m_chips_w_gt_only && filtered_detections->empty() && !had_overlap )
  {
    return false;
  }

  return true;
}


std::string
windowed_trainer::priv
::generate_filename( const std::string& frame_tag, int chip_idx )
{
  std::ostringstream ss;
  ss << frame_tag << "_"
     << std::setw( 5 ) << std::setfill( '0' ) << chip_idx;

  return m_train_directory + div +
         m_chip_subdirectory + div +
         ss.str() + "." + m_chip_format;
}


std::string
windowed_trainer::priv
::frame_tag_for( unsigned fid, const std::string& image_fn )
{
  std::string base = kwiversys::SystemTools::GetFilenameName( image_fn );

  for( auto& c : base )
  {
    if( !std::isalnum( static_cast< unsigned char >( c ) ) &&
        c != '-' && c != '_' )
    {
      c = '_';
    }
  }

  std::ostringstream ss;
  ss << std::setw( 6 ) << std::setfill( '0' ) << fid << "_" << base;
  return ss.str();
}


std::string
windowed_trainer::priv
::manifest_path( const std::string& frame_tag )
{
  return m_train_directory + div +
         m_chip_subdirectory + div +
         frame_tag + ".manifest";
}


bool
windowed_trainer::priv
::load_manifest(
  const std::string& frame_tag,
  std::vector< std::string >& names,
  std::vector< kv::detected_object_set_sptr >& truth )
{
  const std::string mpath = manifest_path( frame_tag );

  if( !kwiversys::SystemTools::FileExists( mpath ) )
  {
    return false;
  }

  std::ifstream ifs( mpath );

  if( !ifs.good() )
  {
    return false;
  }

  std::vector< std::string > tmp_names;
  std::vector< kv::detected_object_set_sptr > tmp_truth;

  std::string line;
  kv::detected_object_set_sptr cur;
  int remaining = 0;

  while( std::getline( ifs, line ) )
  {
    if( line.empty() )
    {
      continue;
    }

    std::istringstream ls( line );
    std::string tag;
    ls >> tag;

    if( tag == "F" )
    {
      std::string fn;
      int ndet = 0;
      ls >> fn >> ndet;

      // Invalidate cache if a referenced file is gone
      if( !kwiversys::SystemTools::FileExists( fn ) )
      {
        return false;
      }

      cur = std::make_shared< kv::detected_object_set >();
      tmp_names.push_back( fn );
      tmp_truth.push_back( cur );
      remaining = ndet;
    }
    else if( tag == "D" && cur && remaining > 0 )
    {
      double minx, miny, maxx, maxy, score;
      std::string cat;
      ls >> cat >> minx >> miny >> maxx >> maxy >> score;

      // Restore spaces encoded as '\x01'
      for( auto& c : cat )
      {
        if( c == '\x01' )
        {
          c = ' ';
        }
      }

      auto dot = std::make_shared< kv::detected_object_type >( cat, score );
      auto dobj = std::make_shared< kv::detected_object >(
        kv::bounding_box_d( minx, miny, maxx, maxy ), score, dot );

      // Optional trailing polygon: P <npts> x1 y1 x2 y2 ...
      std::string ptag;
      if( ls >> ptag && ptag == "P" )
      {
        int npts = 0;
        ls >> npts;

        std::vector< double > poly;
        poly.reserve( npts * 2 );

        for( int pk = 0; pk < npts * 2; ++pk )
        {
          double v;
          if( ls >> v )
          {
            poly.push_back( v );
          }
        }

        if( !poly.empty() )
        {
          dobj->set_flattened_polygon( poly );
        }
      }

      cur->add( dobj );
      --remaining;
    }
    else
    {
      return false;
    }
  }

  for( size_t i = 0; i < tmp_names.size(); ++i )
  {
    names.push_back( tmp_names[i] );
    truth.push_back( tmp_truth[i] );
  }

  return true;
}


kv::category_hierarchy_sptr
windowed_trainer::priv
::labels_without_ignored( kv::category_hierarchy_sptr in )
{
  if( !in || m_ignore_category.empty() || !in->has_class_name( m_ignore_category ) )
  {
    return in;
  }

  // Drop the hard-negative class from the model's output labels
  auto out = std::make_shared< kv::category_hierarchy >();

  for( const auto& name : in->all_class_names() )
  {
    if( name != m_ignore_category )
    {
      out->add_class( name );
    }
  }

  return out;
}


void
windowed_trainer::priv
::write_manifest(
  const std::string& frame_tag,
  const std::vector< std::string >& names,
  const std::vector< kv::detected_object_set_sptr >& truth )
{
  std::ofstream ofs( manifest_path( frame_tag ) );

  if( !ofs.good() )
  {
    return;
  }

  for( size_t i = 0; i < names.size(); ++i )
  {
    kv::detected_object_set_sptr dos = truth[i];
    const size_t ndet = ( dos ? dos->size() : 0 );

    ofs << "F " << names[i] << " " << ndet << "\n";

    if( !dos )
    {
      continue;
    }

    for( auto det = dos->cbegin(); det != dos->cend(); ++det )
    {
      kv::bounding_box_d bb = (*det)->bounding_box();
      double score = (*det)->confidence();
      std::string cat;

      if( (*det)->type() )
      {
        (*det)->type()->get_most_likely( cat );
      }

      // Encode spaces as '\x01' for whitespace-tokenized read-back
      for( auto& c : cat )
      {
        if( std::isspace( static_cast< unsigned char >( c ) ) )
        {
          c = '\x01';
        }
      }

      if( cat.empty() )
      {
        cat = "_";
      }

      // Append the segmentation polygon as: P <npts> x1 y1 x2 y2 ...
      // (kept on the same line; backward compatible with P-less manifests).
      auto poly = (*det)->get_flattened_polygon();

      ofs << "D " << cat << " "
          << bb.min_x() << " " << bb.min_y() << " "
          << bb.max_x() << " " << bb.max_y() << " "
          << score
          << " P " << ( poly.size() / 2 );

      for( double v : poly )
      {
        ofs << " " << v;
      }

      ofs << "\n";
    }
  }
}


void
windowed_trainer::priv
::write_chip_to_disk( const std::string& filename, const kv::image& image )
{
  m_image_io->save( filename,
    kv::image_container_sptr(
      new kv::simple_image_container( image ) ) );
}


} // end namespace viame
