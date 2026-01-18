/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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

#include <vital/util/cpu_timer.h>
#include <vital/algo/image_io.h>
#include <vital/types/image_container.h>

#include <kwiversys/SystemTools.hxx>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstdlib>

namespace viame {

namespace kv = kwiver::vital;

#ifdef WIN32
  const std::string div = "\\";
#else
  const std::string div = "/";
#endif

const std::string windowed_trainer::m_chip_subdirectory = "cached_chips";

// -----------------------------------------------------------------------------
kv::config_block_sptr
windowed_trainer
::get_configuration() const
{
  // Get base config from base class (includes PLUGGABLE_IMPL params)
  kv::config_block_sptr config = kv::algo::train_detector::get_configuration();

  // Common chip settings (shared with detector/refiner)
  config->merge_config( m_settings.chip_config() );

  // Nested algorithm configuration
  kv::get_nested_algo_configuration<kv::algo::image_io>( "image_reader",
    config, m_image_io );
  kv::get_nested_algo_configuration<kv::algo::train_detector>( "trainer",
    config, m_trainer );

  return config;
}


// -----------------------------------------------------------------------------
void
windowed_trainer
::set_configuration_internal( kv::config_block_sptr config_in )
{
  // Merge with defaults
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  // Common chip settings (shared with detector/refiner)
  m_settings.set_chip_config( config );

  if( !c_skip_format )
  {
    // Delete and reset folder contents
    if( kwiversys::SystemTools::FileExists( c_train_directory ) &&
        kwiversys::SystemTools::FileIsDirectory( c_train_directory ) )
    {
      kwiversys::SystemTools::RemoveADirectory( c_train_directory );

#ifndef WIN32
      if( kwiversys::SystemTools::FileExists( c_train_directory ) )
      {
        LOG_ERROR( m_logger, "Unable to delete pre-existing training dir" );
        return;
      }
#endif
    }

    kwiversys::SystemTools::MakeDirectory( c_train_directory );

    if( !m_chip_subdirectory.empty() )
    {
      std::string folder = c_train_directory + div + m_chip_subdirectory;
      kwiversys::SystemTools::MakeDirectory( folder );
    }
  }

  m_detect_small = ( !c_small_action.empty() && c_small_action != "none" );

  kv::algo::image_io_sptr io;
  kv::set_nested_algo_configuration<kv::algo::image_io>( "image_reader", config, io );
  m_image_io = io;

  kv::algo::train_detector_sptr trainer;
  kv::set_nested_algo_configuration<kv::algo::train_detector>( "trainer", config, trainer );
  m_trainer = trainer;
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
::initialize()
{
  m_logger = kv::get_logger( "viame.core.windowed_trainer" );

  // Set trainer-specific defaults (different from detector/refiner)
  m_settings.original_to_chip_size = true;

  // Initialize computed values
  m_synthetic_labels = true;
  m_detect_small = false;
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
    m_labels = object_labels;
    m_synthetic_labels = false;
  }

  std::vector< std::string > filtered_train_names;
  std::vector< kv::detected_object_set_sptr > filtered_train_truth;
  std::vector< std::string > filtered_test_names;
  std::vector< kv::detected_object_set_sptr > filtered_test_truth;

  if( !c_skip_format )
  {
    format_images_from_disk(
      train_image_names, train_groundtruth,
      filtered_train_names, filtered_train_truth );

    format_images_from_disk(
      test_image_names, test_groundtruth,
      filtered_test_names, filtered_test_truth );
  }

  if( m_synthetic_labels )
  {
    kv::category_hierarchy_sptr all_labels =
      std::make_shared< kv::category_hierarchy >();

    for( auto p = m_category_map.begin(); p != m_category_map.end(); p++ )
    {
      all_labels->add_class( p->first );
    }

    m_trainer->add_data_from_disk(
      all_labels,
      filtered_train_names, filtered_train_truth,
      filtered_test_names, filtered_test_truth );
  }
  else
  {
    m_trainer->add_data_from_disk(
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
    m_labels = object_labels;
    m_synthetic_labels = false;
  }

  std::vector< std::string > filtered_train_names;
  std::vector< kv::detected_object_set_sptr > filtered_train_truth;
  std::vector< std::string > filtered_test_names;
  std::vector< kv::detected_object_set_sptr > filtered_test_truth;

  if( !c_skip_format )
  {
    for( unsigned i = 0; i < train_images.size(); ++i )
    {
      kv::image image = train_images[i]->get_image();

      if( c_random_validation > 0.0 &&
          static_cast< double >( rand() ) / RAND_MAX <= c_random_validation )
      {
        format_image_from_memory(
          image, train_groundtruth[i], m_settings.mode,
          filtered_test_names, filtered_test_truth );
      }
      else
      {
        format_image_from_memory(
          image, train_groundtruth[i], m_settings.mode,
          filtered_train_names, filtered_train_truth );
      }
    }
    for( unsigned i = 0; i < test_images.size(); ++i )
    {
      kv::image image = test_images[i]->get_image();

      format_image_from_memory(
        image, test_groundtruth[i], m_settings.mode,
        filtered_test_names, filtered_test_truth );
    }
  }

  m_trainer->add_data_from_disk(
    object_labels,
    filtered_train_names, filtered_train_truth,
    filtered_test_names, filtered_test_truth );
}

void
windowed_trainer
::update_model()
{
  m_trainer->update_model();
}

// -----------------------------------------------------------------------------
void
windowed_trainer
::format_images_from_disk(
  std::vector< std::string > image_names,
  std::vector< kv::detected_object_set_sptr > groundtruth,
  std::vector< std::string >& formatted_names,
  std::vector< kv::detected_object_set_sptr >& formatted_truth )
{
  double negative_ds_factor = -1.0;

  if( c_max_neg_ratio > 0.0 && groundtruth.size() > 10 )
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

      if( current_ratio > c_max_neg_ratio )
      {
        negative_ds_factor = c_max_neg_ratio / current_ratio;
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

    if( m_settings.mode == DISABLED && !c_always_write_image && !c_ensure_standard )
    {
      formatted_names.push_back( image_fn );
      formatted_truth.push_back( groundtruth[fid] );
      continue;
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
        if( c_always_write_image ||
            ( m_settings.original_to_chip_size &&
              ( img_width > m_settings.chip_width ||
                img_height > m_settings.chip_height ) ) ||
            ( c_ensure_standard &&
              ( original_image.depth() != 3 ||
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
      if( img_height <= m_settings.chip_height && img_width <= m_settings.chip_width )
      {
        if( filter_detections_in_roi( groundtruth[fid], image_dims, filtered_truth ) )
        {
          formatted_names.push_back( image_fn );
          formatted_truth.push_back( filtered_truth );
        }
        continue;
      }

      format_mode = MAINTAIN_AR;

      if( ( img_height * img_width ) >= m_settings.chip_adaptive_thresh )
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
windowed_trainer
::format_image_from_memory(
  const kv::image& image,
  kv::detected_object_set_sptr groundtruth,
  const rescale_option format_method,
  std::vector< std::string >& formatted_names,
  std::vector< kv::detected_object_set_sptr >& formatted_truth )
{
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
        if( c_chip_random_factor > 0.0 &&
              static_cast< double >( rand() ) / static_cast<double>( RAND_MAX )
                > c_chip_random_factor )
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
        kv::image cropped_image = crop_image( resized_image, roi );

        double scaled_crop_scale;
        kv::image resized_crop = scale_image_maintaining_ar(
          cropped_image, m_settings.chip_width, m_settings.chip_height,
          m_settings.black_pad, scaled_crop_scale );

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
      double scaled_original_scale;
      kv::image scaled_original = scale_image_maintaining_ar( image,
        m_settings.chip_width, m_settings.chip_height, m_settings.black_pad,
        scaled_original_scale );

      kv::detected_object_set_sptr scaled_original_dets_ptr = groundtruth->clone();
      scaled_original_dets_ptr->scale( scaled_original_scale );

      kv::bounding_box_d roi_box( 0, 0, scaled_original.width(), scaled_original.height() );

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
windowed_trainer
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

    if( det_box.width() < c_min_train_box_length ||
        det_box.height() < c_min_train_box_length )
    {
      return false;
    }

    if( det_box.area() > 0 &&
        overlap.max_x() > overlap.min_x() &&
        overlap.max_y() > overlap.min_y() &&
        overlap.area() / det_box.area() >= c_overlap_required )
    {
      std::string category;

      if( !(*detection)->type() )
      {
        LOG_ERROR( m_logger, "Input detection is missing type category" );
        return false;
      }

      (*detection)->type()->get_most_likely( category );

      if( !c_ignore_category.empty() && category == c_ignore_category )
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

      if( c_min_train_box_edge_dist != 0 &&
          ( min_x <= c_min_train_box_edge_dist ||
            min_y <= c_min_train_box_edge_dist ||
            max_x >= region.width() - c_min_train_box_edge_dist ||
            max_y >= region.height() - c_min_train_box_edge_dist ) )
      {
        return false;
      }

      kv::bounding_box_d bbox( min_x, min_y, max_x, max_y );

      auto odet = (*detection)->clone();
      odet->set_bounding_box( bbox );

      if( m_detect_small && det_box.area() < c_small_box_area )
      {
        if( c_small_action == "remove" )
        {
          continue;
        }
        else if( c_small_action == "skip-chip" )
        {
          return false;
        }

        auto dot_ovr = std::make_shared< kv::detected_object_type >(
          c_small_action, 1.0 );

        odet->set_type( dot_ovr );
      }

      filtered_detections->add( odet );
    }
  }

  return true;
}


std::string
windowed_trainer
::generate_filename( const int len )
{
  static int sample_counter = 0;
  sample_counter++;

  std::ostringstream ss;
  ss << std::setw( len ) << std::setfill( '0' ) << sample_counter;
  std::string s = ss.str();

  return c_train_directory + div +
         m_chip_subdirectory + div +
         s + "." + c_chip_format;
}


void
windowed_trainer
::write_chip_to_disk( const std::string& filename, const kv::image& image )
{
  m_image_io->save( filename,
    kv::image_container_sptr(
      new kv::simple_image_container( image ) ) );
}


} // end namespace viame
