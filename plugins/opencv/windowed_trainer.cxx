/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "windowed_trainer.h"
#include "windowed_utils.h"

#include <vital/algo/algorithm.txx>

#include <plugins/core/utilities_file.h>

#include <vital/util/cpu_timer.h>
#include <vital/algo/image_io.h>

#include <arrows/ocv/image_container.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/bounding_box.h>

#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <cstdint>
#include <cctype>
#include <thread>
#include <mutex>
#include <random>
#include <algorithm>
#include <vector>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

#ifdef WIN32
  const std::string div = "\\";
#else
  const std::string div = "/";
#endif





const std::string windowed_trainer::m_chip_subdirectory = "cached_chips";

// -----------------------------------------------------------------------------
void
windowed_trainer
::initialize()
{
  m_logger = kv::get_logger( "viame.opencv.windowed_trainer" );

  // Set trainer-specific defaults (different from detector/refiner)
  m_settings.original_to_chip_size = true;

  m_synthetic_labels = true;
  m_detect_small = false;
}


// -----------------------------------------------------------------------------
void
windowed_trainer
::set_configuration_internal( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  // Materialise the common chip settings from the pluggable parameters so the
  // rest of this class can keep using a single window_settings object.
  rescale_option_converter conv;
  m_settings.mode = conv.from_string( c_mode );
  m_settings.scale = c_scale;
  m_settings.chip_width = c_chip_width;
  m_settings.chip_height = c_chip_height;
  m_settings.chip_step_width = c_chip_step_width;
  m_settings.chip_step_height = c_chip_step_height;
  m_settings.chip_edge_filter = c_chip_edge_filter;
  m_settings.chip_edge_max_prob = c_chip_edge_max_prob;
  m_settings.chip_adaptive_thresh = c_chip_adaptive_thresh;
  m_settings.batch_size = c_batch_size;
  m_settings.min_detection_dim = c_min_detection_dim;
  m_settings.original_to_chip_size = c_original_to_chip_size;
  m_settings.black_pad = c_black_pad;

  if( !c_skip_format )
  {
    // Delete and reset folder contents, unless reusing a prior chip cache
    if( !c_reuse_cache &&
        kwiversys::SystemTools::FileExists( c_train_directory ) &&
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

    c_trainer->add_data_from_disk(
      all_labels,
      filtered_train_names, filtered_train_truth,
      filtered_test_names, filtered_test_truth );
  }
  else
  {
    c_trainer->add_data_from_disk(
      labels_without_ignored( object_labels ),
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
      cv::Mat image = ocv::image_container::vital_to_ocv(
        train_images[i]->get_image(), ocv::image_container::RGB_COLOR );
      std::mt19937 rng( static_cast< uint64_t >( i ) * 2654435761ull + 1ull );

      if( c_random_validation > 0.0 &&
          static_cast< double >( rand() ) / RAND_MAX <= c_random_validation )
      {
        format_image_from_memory(
          image, train_groundtruth[i], m_settings.mode,
          filtered_test_names, filtered_test_truth,
          "mem_test_" + std::to_string( i ), rng );
      }
      else
      {
        format_image_from_memory(
          image, train_groundtruth[i], m_settings.mode,
          filtered_train_names, filtered_train_truth,
          "mem_train_" + std::to_string( i ), rng );
      }
    }
    for( unsigned i = 0; i < test_images.size(); ++i )
    {
      cv::Mat image = ocv::image_container::vital_to_ocv(
        test_images[i]->get_image(), ocv::image_container::RGB_COLOR );
      std::mt19937 rng( static_cast< uint64_t >( i ) * 2654435761ull + 7ull );

      format_image_from_memory(
        image, test_groundtruth[i], m_settings.mode,
        filtered_test_names, filtered_test_truth,
        "mem_test2_" + std::to_string( i ), rng );
    }
  }

  c_trainer->add_data_from_disk(
    labels_without_ignored( object_labels ),
    filtered_train_names, filtered_train_truth,
    filtered_test_names, filtered_test_truth );
}

std::map<std::string, std::string>
windowed_trainer
::update_model()
{
  std::map<std::string, std::string> nested_output = c_trainer->update_model();

  const std::string algo = "ocv_windowed";
  const std::string nested_prefix = algo + ":detector:";

  std::map<std::string, std::string> output;

  // Re-key nested trainer output so config entries land under
  // the correct .pipe path (e.g. ocv_windowed:detector:netharn:deployed).
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

  // Add ocv_windowed trainer's own config entries
  output["type"] = algo;
  output[algo + ":mode"] = rescale_option_converter().to_string( m_settings.mode );
  output[algo + ":scale"] = std::to_string( m_settings.scale );
  output[algo + ":chip_width"] = std::to_string( m_settings.chip_width );
  output[algo + ":chip_height"] = std::to_string( m_settings.chip_height );
  output[algo + ":chip_step_width"] = std::to_string( m_settings.chip_step_width );
  output[algo + ":chip_step_height"] = std::to_string( m_settings.chip_step_height );
  output[algo + ":chip_adaptive_thresh"] = std::to_string( m_settings.chip_adaptive_thresh );
  output[algo + ":original_to_chip_size"] = m_settings.original_to_chip_size ? "true" : "false";
  output[algo + ":black_pad"] = m_settings.black_pad ? "true" : "false";

  return output;
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

  const unsigned n = static_cast< unsigned >( image_names.size() );

  // Per-frame buffers, merged in frame order for thread independence
  std::vector< std::vector< std::string > > frame_names( n );
  std::vector< std::vector< kv::detected_object_set_sptr > > frame_truth( n );

  unsigned num_threads = ( c_chip_threads > 0 )
    ? static_cast< unsigned >( c_chip_threads )
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
windowed_trainer
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
  if( c_reuse_cache && load_manifest( frame_tag, names, truth ) )
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

  if( m_settings.mode == DISABLED && !c_always_write_image && !c_ensure_standard )
  {
    names.push_back( image_fn );
    truth.push_back( groundtruth[fid] );
    write_manifest( frame_tag, names, truth );
    return;
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

    vital_image = c_image_reader->load( image_fn );

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
      if( c_always_write_image ||
          ( m_settings.original_to_chip_size &&
            ( original_image.cols > m_settings.chip_width ||
              original_image.rows > m_settings.chip_height ) ) ||
          ( c_ensure_standard &&
            ( original_image.channels() != 3 ||
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
    if( original_image.rows <= m_settings.chip_height &&
        original_image.cols <= m_settings.chip_width )
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

    if( ( original_image.rows * original_image.cols ) >= m_settings.chip_adaptive_thresh )
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
windowed_trainer
::format_image_from_memory(
  const cv::Mat& image,
  kv::detected_object_set_sptr groundtruth,
  const rescale_option format_method,
  std::vector< std::string >& formatted_names,
  std::vector< kv::detected_object_set_sptr >& formatted_truth,
  const std::string& frame_tag,
  std::mt19937& rng )
{
  int chip_idx = 0;
  std::uniform_real_distribution< double > unif( 0.0, 1.0 );
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
      std::string img_file = generate_filename( frame_tag, chip_idx++ );
      write_chip_to_disk( img_file, resized_image );

      formatted_names.push_back( img_file );
      formatted_truth.push_back( filtered_truth );
    }
  }
  else
  {
    int annotated_chips = 0;
    std::vector< cv::Rect > background_rois;

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
        if( c_chip_random_factor > 0.0 &&
              unif( rng ) > c_chip_random_factor )
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

        cv::Rect roi( i, j, cw, ch );

        kv::bounding_box_d roi_box( i, j, i + m_settings.chip_width,
          j + m_settings.chip_height );

        bool overlapped = false;

        if( filter_detections_in_roi( scaled_groundtruth, roi_box,
              filtered_truth, &overlapped ) )
        {
          cv::Mat cropped_image = resized_image( roi );
          cv::Mat resized_crop;

          scale_image_maintaining_ar( cropped_image,
            resized_crop, m_settings.chip_width, m_settings.chip_height,
            m_settings.black_pad );

          std::string img_file = generate_filename( frame_tag, chip_idx++ );
          write_chip_to_disk( img_file, resized_crop );

          formatted_names.push_back( img_file );
          formatted_truth.push_back( filtered_truth );
          ++annotated_chips;
        }
        else if( c_background_chip_ratio > 0.0 && !overlapped )
        {
          background_rois.push_back( roi );
        }
      }
    }

    // Sample background chips proportional to annotated chips
    if( c_background_chip_ratio > 0.0 &&
        annotated_chips > 0 && !background_rois.empty() )
    {
      int target = static_cast< int >(
        c_background_chip_ratio * annotated_chips + 0.5 );
      target = std::min( target, static_cast< int >( background_rois.size() ) );

      std::shuffle( background_rois.begin(), background_rois.end(), rng );

      for( int n = 0; n < target; ++n )
      {
        cv::Mat cropped_image = resized_image( background_rois[n] );
        cv::Mat resized_crop;

        scale_image_maintaining_ar( cropped_image,
          resized_crop, m_settings.chip_width, m_settings.chip_height,
          m_settings.black_pad );

        std::string img_file = generate_filename( frame_tag, chip_idx++ );
        write_chip_to_disk( img_file, resized_crop );

        formatted_names.push_back( img_file );
        formatted_truth.push_back( std::make_shared< kv::detected_object_set >() );
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
        std::string img_file = generate_filename( frame_tag, chip_idx++ );
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

      if( !c_ignore_category.empty() && category == c_ignore_category )
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

  // Drop only chips with no annotation at all (hard-negative chips kept)
  if( c_chips_w_gt_only && filtered_detections->empty() && !had_overlap )
  {
    return false;
  }

  return true;
}


std::string
windowed_trainer
::generate_filename( const std::string& frame_tag, int chip_idx )
{
  std::ostringstream ss;
  ss << frame_tag << "_"
     << std::setw( 5 ) << std::setfill( '0' ) << chip_idx;

  return c_train_directory + div +
         m_chip_subdirectory + div +
         ss.str() + "." + c_chip_format;
}


std::string
windowed_trainer
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
windowed_trainer
::manifest_path( const std::string& frame_tag )
{
  return c_train_directory + div +
         m_chip_subdirectory + div +
         frame_tag + ".manifest";
}


bool
windowed_trainer
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
windowed_trainer
::labels_without_ignored( kv::category_hierarchy_sptr in )
{
  if( !in || c_ignore_category.empty() || !in->has_class_name( c_ignore_category ) )
  {
    return in;
  }

  // Drop the hard-negative class from the model's output labels
  auto out = std::make_shared< kv::category_hierarchy >();

  for( const auto& name : in->all_class_names() )
  {
    if( name != c_ignore_category )
    {
      out->add_class( name );
    }
  }

  return out;
}


void
windowed_trainer
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
windowed_trainer
::write_chip_to_disk( const std::string& filename, const cv::Mat& image )
{
  c_image_reader->save( filename,
    kv::image_container_sptr(
      new ocv::image_container( image,
        ocv::image_container::RGB_COLOR ) ) );
}


} // end namespace viame
