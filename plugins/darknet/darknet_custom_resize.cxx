/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "darknet_custom_resize.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <exception>

namespace viame {

// =============================================================================

double
scale_image_maintaining_ar( const cv::Mat& src, cv::Mat& dst,
                            int width, int height )
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

  dst.create( height, width, src.type() );
  dst.setTo( 0 );

  cv::Rect roi( 0, 0, resized.cols, resized.rows );
  cv::Mat aoi( dst, roi );

  resized.copyTo( aoi );
  return scale;
}


// -----------------------------------------------------------------------------
double
format_image( const cv::Mat& src, cv::Mat& dst, std::string option,
              double scale_factor, int width, int height )
{
  double scale = 1.0;

  if( option == "maintain_ar" )
  {
    scale = scale_image_maintaining_ar( src, dst, width, height );
  }
  else if( option == "chip" || option == "scale" ||
           option == "chip_and_original" )
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
    throw std::runtime_error( "Invalid resize option: " + option );
  }

  return scale;
}


} // end namespace
