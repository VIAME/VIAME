/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "windowed_utils.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <exception>

namespace viame {

// =============================================================================
// OpenCV-specific image processing functions
// Note: window_settings, windowed_region_prop constructors, and all detection
// manipulation functions are now in core/windowed_utils.cxx
// =============================================================================

double
scale_image_maintaining_ar( const cv::Mat& src, cv::Mat& dst,
                            int width, int height, bool pad )
{
  double scale = 1.0;

  if( src.rows == height && src.cols == width )
  {
    dst = src;
    return scale;
  }

  double original_height = static_cast< double >( src.rows );
  double original_width = static_cast< double >( src.cols );

  if( original_height > height )
  {
    scale = height / original_height;
  }
  if( original_width > width )
  {
    scale = std::min( scale, width / original_width );
  }

  cv::Mat resized;
  cv::resize( src, resized, cv::Size(), scale, scale );

  if( pad )
  {
    dst.create( height, width, src.type() );
    dst.setTo( 0 );

    cv::Rect roi( 0, 0, resized.cols, resized.rows );
    cv::Mat aoi( dst, roi );

    resized.copyTo( aoi );
  }
  else
  {
    dst = resized;
  }

  return scale;
}

double
format_image( const cv::Mat& src, cv::Mat& dst, rescale_option option,
              double scale_factor, int width, int height, bool pad )
{
  double scale = 1.0;

  if( option == MAINTAIN_AR )
  {
    scale = scale_image_maintaining_ar( src, dst, width, height, pad );
  }
  else if( option == CHIP || option == SCALE ||
           option == CHIP_AND_ORIGINAL )
  {
    if( scale_factor == 1.0 )
    {
      dst = src;
    }
    else
    {
      cv::resize( src, dst, cv::Size(), scale_factor, scale_factor );
      scale = scale_factor;
    }
  }
  else
  {
    rescale_option_converter conv;
    throw std::runtime_error( "Invalid resize option: " + conv.to_string( option ) );
  }

  return scale;
}

// -----------------------------------------------------------------------------
void
prepare_image_regions(
  const cv::Mat& image,
  const window_settings& settings,
  std::vector< cv::Mat >& regions_to_process,
  std::vector< windowed_region_prop >& region_properties )
{
  // Clear output vectors
  regions_to_process.clear();
  region_properties.clear();

  // Determine the processing mode
  rescale_option mode = settings.mode;

  if( mode == ADAPTIVE )
  {
    if( ( image.rows * image.cols ) >= settings.chip_adaptive_thresh )
    {
      mode = CHIP_AND_ORIGINAL;
    }
    else if( settings.original_to_chip_size )
    {
      mode = MAINTAIN_AR;
    }
    else
    {
      mode = DISABLED;
    }
  }

  // Resize image if enabled
  cv::Mat resized_image;
  double scale_factor = 1.0;

  if( mode != DISABLED )
  {
    scale_factor = format_image( image, resized_image,
      ( mode == ORIGINAL_AND_RESIZED ? SCALE : mode ),
      settings.scale, settings.chip_width, settings.chip_height );
  }
  else
  {
    resized_image = image;
  }

  image_rect original_dims( 0, 0, image.cols, image.rows );

  // Create regions based on mode
  if( mode == ORIGINAL_AND_RESIZED )
  {
    cv::Mat scaled_original;

    if( image.rows <= settings.chip_height && image.cols <= settings.chip_width )
    {
      regions_to_process.push_back( image );
      region_properties.push_back( windowed_region_prop( original_dims, 1.0 ) );
    }
    else
    {
      if( ( image.rows * image.cols ) >= settings.chip_adaptive_thresh )
      {
        regions_to_process.push_back( resized_image );
        region_properties.push_back( windowed_region_prop( original_dims, 1.0 / scale_factor ) );
      }

      double scaled_original_scale = scale_image_maintaining_ar( image,
        scaled_original, settings.chip_width, settings.chip_height, settings.black_pad );

      regions_to_process.push_back( scaled_original );
      region_properties.push_back( windowed_region_prop( original_dims, 1.0 / scaled_original_scale ) );
    }
  }
  else if( mode != CHIP && mode != CHIP_AND_ORIGINAL )
  {
    regions_to_process.push_back( resized_image );
    region_properties.push_back( windowed_region_prop( original_dims, 1.0 / scale_factor ) );
  }
  else
  {
    // Chip up scaled image
    for( int li = 0;
         li < resized_image.cols - settings.chip_width + settings.chip_step_width;
         li += settings.chip_step_width )
    {
      int ti = std::min( li + settings.chip_width, resized_image.cols );

      for( int lj = 0;
           lj < resized_image.rows - settings.chip_height + settings.chip_step_height;
           lj += settings.chip_step_height )
      {
        int tj = std::min( lj + settings.chip_height, resized_image.rows );

        if( tj-lj < 0 || ti-li < 0 )
        {
          continue;
        }

        cv::Rect resized_roi( li, lj, ti-li, tj-lj );
        image_rect original_roi(
          static_cast< int >( li / scale_factor ),
          static_cast< int >( lj / scale_factor ),
          static_cast< int >( (ti-li) / scale_factor ),
          static_cast< int >( (tj-lj) / scale_factor ) );

        cv::Mat cropped_chip = resized_image( resized_roi );
        cv::Mat scaled_crop;

        double scaled_crop_scale = scale_image_maintaining_ar(
          cropped_chip, scaled_crop, settings.chip_width, settings.chip_height,
          settings.black_pad );

        regions_to_process.push_back( scaled_crop );

        region_properties.push_back(
          windowed_region_prop( original_roi,
            settings.chip_edge_filter,
            ( li + settings.chip_step_width ) >=
              ( resized_image.cols - settings.chip_width + settings.chip_step_width ),
            ( lj + settings.chip_step_height ) >=
              ( resized_image.rows - settings.chip_height + settings.chip_step_height ),
            1.0 / scaled_crop_scale,
            li, lj,
            1.0 / scale_factor ) );
      }
    }

    // Extract full sized image chip if enabled
    if( mode == CHIP_AND_ORIGINAL )
    {
      cv::Mat scaled_original;

      if( settings.original_to_chip_size )
      {
        double scaled_original_scale = scale_image_maintaining_ar( image,
          scaled_original, settings.chip_width, settings.chip_height, settings.black_pad );

        regions_to_process.push_back( scaled_original );
        region_properties.push_back( windowed_region_prop( original_dims, 1.0 / scaled_original_scale ) );
      }
      else
      {
        regions_to_process.push_back( image );
        region_properties.push_back( windowed_region_prop( original_dims, 1.0 ) );
      }
    }
  }
}

} // end namespace viame
