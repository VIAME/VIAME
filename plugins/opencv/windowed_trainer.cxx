/*ckwg +29
 * Copyright 2019-2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "windowed_trainer.h"

#include <vital/algo/algorithm.txx>

#include "windowed_utils.h"

#include <vital/util/cpu_timer.h>
#include <vital/algo/image_io.h>

#include <arrows/ocv/image_container.h>

#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

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

  // Helper functions
  void format_images_from_disk(
    std::vector< std::string > image_names,
    std::vector< kv::detected_object_set_sptr > groundtruth,
    std::vector< std::string >& formatted_names,
    std::vector< kv::detected_object_set_sptr >& formatted_truth );

  void format_image_from_memory(
    const cv::Mat& image,
    kv::detected_object_set_sptr groundtruth,
    const rescale_option format_method,
    std::vector< std::string >& formatted_names,
    std::vector< kv::detected_object_set_sptr >& formatted_truth );

  bool filter_detections_in_roi(
    kv::detected_object_set_sptr all_detections,
    kv::bounding_box_d region,
    kv::detected_object_set_sptr& filt_detections );

  std::string generate_filename( const int len = 10 );

  void write_chip_to_disk( const std::string& filename, const cv::Mat& image );

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
  attach_logger( "viame.opencv.windowed_trainer" );

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

  kv::get_nested_algo_configuration<kv::algo::image_io>( "image_reader",
    config, d->m_image_io );
  kv::get_nested_algo_configuration<kv::algo::train_detector>( "trainer",
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

  if( !d->m_skip_format )
  {
    // Delete and reset folder contents
    if( kwiversys::SystemTools::FileExists( d->m_train_directory ) &&
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
  kv::set_nested_algo_configuration<kv::algo::image_io>( "image_reader", config, io );
  d->m_image_io = io;

  kv::algo::train_detector_sptr trainer;
  kv::set_nested_algo_configuration<kv::algo::train_detector>( "trainer", config, trainer );
  d->m_trainer = trainer;
}


// -----------------------------------------------------------------------------
bool
windowed_trainer
::check_configuration( kv::config_block_sptr config ) const
{
  return kv::check_nested_algo_configuration<kv::algo::image_io>(
     "image_reader", config )
   && kv::check_nested_algo_configuration<kv::algo::train_detector>(
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
      object_labels,
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
      cv::Mat image = ocv::image_container::vital_to_ocv(
        train_images[i]->get_image(), ocv::image_container::RGB_COLOR );

      if( d->m_random_validation > 0.0 &&
          static_cast< double >( rand() ) / RAND_MAX <= d->m_random_validation )
      {
        d->format_image_from_memory(
          image, train_groundtruth[i], d->m_settings.mode,
          filtered_test_names, filtered_test_truth );
      }
      else
      {
        d->format_image_from_memory(
          image, train_groundtruth[i], d->m_settings.mode,
          filtered_train_names, filtered_train_truth );
      }
    }
    for( unsigned i = 0; i < test_images.size(); ++i )
    {
      cv::Mat image = ocv::image_container::vital_to_ocv(
        test_images[i]->get_image(), ocv::image_container::RGB_COLOR );

      d->format_image_from_memory(
        image, test_groundtruth[i], d->m_settings.mode,
        filtered_test_names, filtered_test_truth );
    }
  }

  d->m_trainer->add_data_from_disk(
    object_labels,
    filtered_train_names, filtered_train_truth,
    filtered_test_names, filtered_test_truth );
}

void
windowed_trainer
::update_model()
{
  d->m_trainer->update_model();
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

  for( unsigned fid = 0; fid < image_names.size(); ++fid )
  {
    if( negative_ds_factor > 0.0 &&
        ( !groundtruth[fid] || groundtruth[fid]->empty() ) &&
        static_cast< double >( rand() ) / RAND_MAX > negative_ds_factor )
    {
      continue;
    }

    const std::string image_fn = image_names[fid];

    if( m_settings.mode == DISABLED && !m_always_write_image && !m_ensure_standard )
    {
      formatted_names.push_back( image_fn );
      formatted_truth.push_back( groundtruth[fid] );
      continue;
    }

    // Scale and break up image according to settings
    kv::image_container_sptr vital_image;
    kv::bounding_box_d image_dims;
    cv::Mat original_image;
    kv::detected_object_set_sptr filtered_truth;

    rescale_option format_mode = m_settings.mode;
    std::string ext = image_fn.substr( image_fn.find_last_of( "." ) + 1 );

    try
    {
      LOG_INFO( m_logger, "Loading image: " << image_fn );

      vital_image = m_image_io->load( image_fn );

      original_image = ocv::image_container::vital_to_ocv(
        vital_image->get_image(), ocv::image_container::RGB_COLOR );

      image_dims = kv::bounding_box_d( 0, 0,
        original_image.cols, original_image.rows );
    }
    catch( const kv::vital_exception& e )
    {
      LOG_ERROR( m_logger, "Caught exception reading image: " << e.what() );
      return;
    }

    // Early exit don't need to read all images every iteration
    if( format_mode == ADAPTIVE )
    {
      if( ( original_image.rows * original_image.cols ) < m_settings.chip_adaptive_thresh )
      {
        if( m_always_write_image ||
            ( m_settings.original_to_chip_size &&
              ( original_image.cols > m_settings.chip_width ||
                original_image.rows > m_settings.chip_height ) ) ||
            ( m_ensure_standard &&
              ( original_image.channels() != 3 ||
               !( ext == "jpg" || ext == "png" || ext == "jpeg" ) ) ) )
        {
          format_mode = MAINTAIN_AR;
        }
        else
        {
          if( filter_detections_in_roi( groundtruth[fid], image_dims, filtered_truth ) )
          {
            formatted_names.push_back( image_fn );
            formatted_truth.push_back( filtered_truth );
          }
          continue;
        }
      }
      else
      {
        format_mode = CHIP_AND_ORIGINAL;
      }
    }
    else if( format_mode == ORIGINAL_AND_RESIZED )
    {
      if( original_image.rows <= m_settings.chip_height &&
          original_image.cols <= m_settings.chip_width )
      {
        if( filter_detections_in_roi( groundtruth[fid], image_dims, filtered_truth ) )
        {
          formatted_names.push_back( image_fn );
          formatted_truth.push_back( filtered_truth );
        }
        continue;
      }

      format_mode = MAINTAIN_AR;

      if( ( original_image.rows * original_image.cols ) >= m_settings.chip_adaptive_thresh )
      {
        if( filter_detections_in_roi( groundtruth[fid], image_dims, filtered_truth ) )
        {
          formatted_names.push_back( image_fn );
          formatted_truth.push_back( filtered_truth );
        }
      }
    }

    // Format image and write new ones to disk
    format_image_from_memory(
      original_image, groundtruth[fid], format_mode,
      formatted_names, formatted_truth );
  }
}

void
windowed_trainer::priv
::format_image_from_memory(
  const cv::Mat& image,
  kv::detected_object_set_sptr groundtruth,
  const rescale_option format_method,
  std::vector< std::string >& formatted_names,
  std::vector< kv::detected_object_set_sptr >& formatted_truth )
{
  cv::Mat resized_image;
  kv::detected_object_set_sptr scaled_groundtruth = groundtruth->clone();
  kv::detected_object_set_sptr filtered_truth;

  double resized_scale = 1.0;

  if( format_method != DISABLED )
  {
    resized_scale = format_image( image, resized_image,
      format_method, m_settings.scale, m_settings.chip_width,
      m_settings.chip_height, m_settings.black_pad );

    scaled_groundtruth->scale( resized_scale );
  }
  else
  {
    resized_image = image;
    scaled_groundtruth = groundtruth;
  }

  if( format_method != CHIP && format_method != CHIP_AND_ORIGINAL )
  {
    kv::bounding_box_d roi_box( 0, 0, resized_image.cols, resized_image.rows );

    if( filter_detections_in_roi( scaled_groundtruth, roi_box, filtered_truth ) )
    {
      std::string img_file = generate_filename();
      write_chip_to_disk( img_file, resized_image );

      formatted_names.push_back( img_file );
      formatted_truth.push_back( filtered_truth );
    }
  }
  else
  {
    // Chip up and process scaled image
    for( int i = 0;
         i < resized_image.cols - m_settings.chip_width + m_settings.chip_step_width;
         i += m_settings.chip_step_width )
    {
      int cw = i + m_settings.chip_width;

      if( cw > resized_image.cols )
      {
        cw = resized_image.cols - i;
      }
      else
      {
        cw = m_settings.chip_width;
      }

      for( int j = 0;
           j < resized_image.rows - m_settings.chip_height + m_settings.chip_step_height;
           j += m_settings.chip_step_height )
      {
        // random downsampling
        if( m_chip_random_factor > 0.0 &&
              static_cast< double >( rand() ) / static_cast<double>( RAND_MAX )
                > m_chip_random_factor )
        {
          continue;
        }

        int ch = j + m_settings.chip_height;

        if( ch > resized_image.rows )
        {
          ch = resized_image.rows - j;
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

        cv::Mat cropped_image = resized_image( cv::Rect( i, j, cw, ch ) );
        cv::Mat resized_crop;

        scale_image_maintaining_ar( cropped_image,
          resized_crop, m_settings.chip_width, m_settings.chip_height,
          m_settings.black_pad );

        kv::bounding_box_d roi_box( i, j, i + m_settings.chip_width,
          j + m_settings.chip_height );

        if( filter_detections_in_roi( scaled_groundtruth, roi_box, filtered_truth ) )
        {
          std::string img_file = generate_filename();
          write_chip_to_disk( img_file, resized_crop );

          formatted_names.push_back( img_file );
          formatted_truth.push_back( filtered_truth );
        }
      }
    }

    // Process full sized image if enabled
    if( format_method == CHIP_AND_ORIGINAL )
    {
      cv::Mat scaled_original;

      double scaled_original_scale = scale_image_maintaining_ar( image,
        scaled_original, m_settings.chip_width, m_settings.chip_height,
        m_settings.black_pad );

      kv::detected_object_set_sptr scaled_original_dets_ptr = groundtruth->clone();
      scaled_original_dets_ptr->scale( scaled_original_scale );

      kv::bounding_box_d roi_box( 0, 0, scaled_original.cols, scaled_original.rows );

      if( filter_detections_in_roi( scaled_original_dets_ptr, roi_box, filtered_truth ) )
      {
        std::string img_file = generate_filename();
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
  kv::detected_object_set_sptr& filtered_detections )
{
  auto ie = all_detections->cend();

  filtered_detections = std::make_shared< kv::detected_object_set >();

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

  return true;
}


std::string
windowed_trainer::priv
::generate_filename( const int len )
{
  static int sample_counter = 0;
  sample_counter++;

  std::ostringstream ss;
  ss << std::setw( len ) << std::setfill( '0' ) << sample_counter;
  std::string s = ss.str();

  return m_train_directory + div +
         m_chip_subdirectory + div +
         s + "." + m_chip_format;
}


void
windowed_trainer::priv
::write_chip_to_disk( const std::string& filename, const cv::Mat& image )
{
  m_image_io->save( filename,
    kv::image_container_sptr(
      new ocv::image_container( image,
        ocv::image_container::RGB_COLOR ) ) );
}


} // end namespace viame
