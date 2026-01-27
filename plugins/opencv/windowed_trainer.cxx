/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "windowed_trainer.h"

#include <vital/algo/algorithm.txx>

#include "windowed_utils.h"

#include <vital/util/cpu_timer.h>
#include <vital/algo/image_io.h>

#include <arrows/ocv/image_container.h>

#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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
// -----------------------------------------------------------------------------
void
windowed_trainer
::format_images_from_disk(
  const window_settings& settings,
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

    if( settings.mode == DISABLED && !c_always_write_image && !c_ensure_standard )
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

    rescale_option format_mode = settings.mode;
    std::string ext = image_fn.substr( image_fn.find_last_of( "." ) + 1 );

    try
    {
      LOG_INFO( logger(), "Loading image: " << image_fn );

      vital_image = c_image_reader->load( image_fn );

      original_image = ocv::image_container::vital_to_ocv(
        vital_image->get_image(), ocv::image_container::RGB_COLOR );

      image_dims = kv::bounding_box_d( 0, 0,
        original_image.cols, original_image.rows );
    }
    catch( const kv::vital_exception& e )
    {
      LOG_ERROR( logger(), "Caught exception reading image: " << e.what() );
      return;
    }

    // Early exit don't need to read all images every iteration
    if( format_mode == ADAPTIVE )
    {
      if( ( original_image.rows * original_image.cols ) < settings.chip_adaptive_thresh )
      {
        if( c_always_write_image ||
            ( settings.original_to_chip_size &&
              ( original_image.cols > settings.chip_width ||
                original_image.rows > settings.chip_height ) ) ||
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
      if( original_image.rows <= settings.chip_height &&
          original_image.cols <= settings.chip_width )
      {
        if( filter_detections_in_roi( groundtruth[fid], image_dims, filtered_truth ) )
        {
          formatted_names.push_back( image_fn );
          formatted_truth.push_back( filtered_truth );
        }
        continue;
      }

      format_mode = MAINTAIN_AR;

      if( ( original_image.rows * original_image.cols ) >= settings.chip_adaptive_thresh )
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
      settings, original_image, groundtruth[fid], format_mode,
      formatted_names, formatted_truth );
  }
}

void
windowed_trainer
::format_image_from_memory(
  const window_settings& settings,
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
      format_method, settings.scale, settings.chip_width,
      settings.chip_height, settings.black_pad );

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
         i < resized_image.cols - settings.chip_width + settings.chip_step_width;
         i += settings.chip_step_width )
    {
      int cw = i + settings.chip_width;

      if( cw > resized_image.cols )
      {
        cw = resized_image.cols - i;
      }
      else
      {
        cw = settings.chip_width;
      }

      for( int j = 0;
           j < resized_image.rows - settings.chip_height + settings.chip_step_height;
           j += settings.chip_step_height )
      {
        // random downsampling
        if( c_chip_random_factor > 0.0 &&
              static_cast< double >( rand() ) / static_cast<double>( RAND_MAX )
                > c_chip_random_factor )
        {
          continue;
        }

        int ch = j + settings.chip_height;

        if( ch > resized_image.rows )
        {
          ch = resized_image.rows - j;
        }
        else
        {
          ch = settings.chip_height;
        }

        // Only necessary in a few circumstances when chip_step exceeds image size.
        if( ch < 0 || cw < 0 )
        {
          continue;
        }

        cv::Mat cropped_image = resized_image( cv::Rect( i, j, cw, ch ) );
        cv::Mat resized_crop;

        scale_image_maintaining_ar( cropped_image,
          resized_crop, settings.chip_width, settings.chip_height,
          settings.black_pad );

        kv::bounding_box_d roi_box( i, j, i + settings.chip_width,
          j + settings.chip_height );

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
        scaled_original, settings.chip_width, settings.chip_height,
        settings.black_pad );

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
windowed_trainer
::filter_detections_in_roi(
  kv::detected_object_set_sptr all_detections,
  kv::bounding_box_d region,
  kv::detected_object_set_sptr& filtered_detections )
{
  auto ie = all_detections->cend();

  filtered_detections = std::make_shared< kv::detected_object_set >();

  bool detect_small = ( !c_small_action.empty() && c_small_action != "none" );

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
        LOG_ERROR( logger(), "Input detection is missing type category" );
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
        LOG_WARN( logger(), "Ignoring unlisted class " << category );
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

      if( detect_small && det_box.area() < c_small_box_area )
      {
        if( c_small_action == "remove" )
        {
          continue;
        }
        else if( c_small_action == "skip-chip" )
        {
          return false;
        }
        else
        {
          odet->set_type( std::make_shared< kv::detected_object_type >( c_small_action, 1.0 ) );
        }
      }
      else
      {
        odet->set_type( std::make_shared< kv::detected_object_type >( category, 1.0 ) );
      }

      filtered_detections->add( odet );
    }
  }

  if( c_chips_w_gt_only && filtered_detections->empty() )
  {
    return false;
  }

  return true;
}

std::string
windowed_trainer
::generate_filename( const int len )
{
  static const char alphanum[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

  static bool seeded = false;

  if( !seeded )
  {
    srand( ( unsigned ) time( NULL ) + getpid() );
    seeded = true;
  }

  std::string str;

  for ( int i = 0; i < len; ++i )
  {
    str += alphanum[ rand() % ( sizeof( alphanum ) - 1 ) ];
  }

  return c_train_directory + div + m_chip_subdirectory + div + str + "." + c_chip_format;
}

void
windowed_trainer
::write_chip_to_disk( const std::string& filename, const cv::Mat& image )
{
  cv::imwrite( filename, image );
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
    // Reconstruct settings
    window_settings settings;
    rescale_option_converter conv;
    settings.mode = conv.from_string( c_mode );
    settings.scale = c_scale;
    settings.chip_width = c_chip_width;
    settings.chip_height = c_chip_height;
    settings.chip_step_width = c_chip_step_width;
    settings.chip_step_height = c_chip_step_height;
    settings.chip_edge_filter = c_chip_edge_filter;
    settings.chip_edge_max_prob = c_chip_edge_max_prob;
    settings.chip_adaptive_thresh = c_chip_adaptive_thresh;
    settings.batch_size = c_batch_size;
    settings.min_detection_dim = c_min_detection_dim;
    settings.original_to_chip_size = c_original_to_chip_size;
    settings.black_pad = c_black_pad;

    // Derived detect_small
    bool detect_small = ( !c_small_action.empty() && c_small_action != "none" );

    // Ensure directories exist
    if( kwiversys::SystemTools::FileExists( c_train_directory ) &&
        kwiversys::SystemTools::FileIsDirectory( c_train_directory ) )
    {
      kwiversys::SystemTools::RemoveADirectory( c_train_directory );
    }
    kwiversys::SystemTools::MakeDirectory( c_train_directory );
    if( !m_chip_subdirectory.empty() )
    {
      std::string folder = c_train_directory + div + m_chip_subdirectory;
      kwiversys::SystemTools::MakeDirectory( folder );
    }

    format_images_from_disk(
      settings,
      train_image_names, train_groundtruth,
      filtered_train_names, filtered_train_truth );

    format_images_from_disk(
      settings,
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
    // Reconstruct settings
    window_settings settings;
    rescale_option_converter conv;
    settings.mode = conv.from_string( c_mode );
    settings.scale = c_scale;
    settings.chip_width = c_chip_width;
    settings.chip_height = c_chip_height;
    settings.chip_step_width = c_chip_step_width;
    settings.chip_step_height = c_chip_step_height;
    settings.chip_edge_filter = c_chip_edge_filter;
    settings.chip_edge_max_prob = c_chip_edge_max_prob;
    settings.chip_adaptive_thresh = c_chip_adaptive_thresh;
    settings.batch_size = c_batch_size;
    settings.min_detection_dim = c_min_detection_dim;
    settings.original_to_chip_size = c_original_to_chip_size;
    settings.black_pad = c_black_pad;

    // Ensure directories exist (duplicated logic from add_data_from_disk, could be helper)
    if( kwiversys::SystemTools::FileExists( c_train_directory ) &&
        kwiversys::SystemTools::FileIsDirectory( c_train_directory ) )
    {
      kwiversys::SystemTools::RemoveADirectory( c_train_directory );
    }
    kwiversys::SystemTools::MakeDirectory( c_train_directory );
    if( !m_chip_subdirectory.empty() )
    {
      std::string folder = c_train_directory + div + m_chip_subdirectory;
      kwiversys::SystemTools::MakeDirectory( folder );
    }

    for( unsigned i = 0; i < train_images.size(); ++i )
    {
      cv::Mat image = ocv::image_container::vital_to_ocv(
        train_images[i]->get_image(), ocv::image_container::RGB_COLOR );

      if( c_random_validation > 0.0 &&
          static_cast< double >( rand() ) / RAND_MAX <= c_random_validation )
      {
        format_image_from_memory(
          settings, image, train_groundtruth[i], settings.mode,
          filtered_test_names, filtered_test_truth );
      }
      else
      {
        format_image_from_memory(
          settings, image, train_groundtruth[i], settings.mode,
          filtered_train_names, filtered_train_truth );
      }
    }
    for( unsigned i = 0; i < test_images.size(); ++i )
    {
      cv::Mat image = ocv::image_container::vital_to_ocv(
        test_images[i]->get_image(), ocv::image_container::RGB_COLOR );

      format_image_from_memory(
        settings, image, test_groundtruth[i], settings.mode,
        filtered_test_names, filtered_test_truth );
    }
  }

  c_trainer->add_data_from_disk(
    object_labels,
    filtered_train_names, filtered_train_truth,
    filtered_test_names, filtered_test_truth );
}

std::map<std::string, std::string>
windowed_trainer
::update_model()
{
  std::map<std::string, std::string> output = c_trainer->update_model();

  const std::string algo = "ocv_windowed";

  output["type"] = algo;
  output[algo + ":mode"] = c_mode;
  output[algo + ":scale"] = std::to_string( c_scale );
  output[algo + ":chip_width"] = std::to_string( c_chip_width );
  output[algo + ":chip_height"] = std::to_string( c_chip_height );
  output[algo + ":chip_step_width"] = std::to_string( c_chip_step_width );
  output[algo + ":chip_step_height"] = std::to_string( c_chip_step_height );
  output[algo + ":chip_adaptive_thresh"] = std::to_string( c_chip_adaptive_thresh );
  output[algo + ":original_to_chip_size"] = c_original_to_chip_size ? "true" : "false";
  output[algo + ":black_pad"] = c_black_pad ? "true" : "false";

  return output;
}

// -----------------------------------------------------------------------------

} // end namespace viame
