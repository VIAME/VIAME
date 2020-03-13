/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#include "windowed_detector.h"
#include "windowed_detector_resize.h"

#include <vital/util/wall_timer.h>
#include <vital/exceptions/io.h>
#include <vital/config/config_block_formatter.h>

#include <arrows/ocv/image_container.h>
#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <sstream>
#include <exception>
#include <limits>

namespace kwiver {
namespace arrows {
namespace ocv {

// =============================================================================
class windowed_detector::priv
{
public:
  priv()
    : m_mode( "disabled" )
    , m_scale( 1.0 )
    , m_chip_width( 1000 )
    , m_chip_height( 1000 )
    , m_chip_step_width( 500 )
    , m_chip_step_height( 500 )
    , m_chip_edge_filter( 0 )
    , m_chip_adaptive_thresh( 2000000 )
    , m_batch_size( 1 )
    , m_min_detection_dim( 2 )
    , m_original_to_chip_size( false )
    , m_black_pad( false )
  {}

  ~priv() {}

  // Items from the config
  std::string m_mode;
  double m_scale;
  int m_chip_width;
  int m_chip_height;
  int m_chip_step_width;
  int m_chip_step_height;
  int m_chip_edge_filter;
  int m_chip_adaptive_thresh;
  int m_batch_size;
  int m_min_detection_dim;
  bool m_original_to_chip_size;
  bool m_black_pad;

  // Helper functions
  struct region_info
  {
    explicit region_info( cv::Rect r, double s1 )
     : original_roi( r ), edge_filter( 0 ),
       scale1( s1 ), shiftx( 0 ), shifty( 0 ), scale2( 1.0 )
    {}

    explicit region_info( cv::Rect r, int ef,
      double s1, int sx, int sy, double s2 )
     : original_roi( r ), edge_filter( ef ),
       scale1( s1 ), shiftx( sx ), shifty( sy ), scale2( s2 )
    {}

    cv::Rect original_roi;
    int edge_filter;
    double scale1;
    int shiftx, shifty;
    double scale2;
  };

  vital::detected_object_set_sptr scale_detections(
    const vital::detected_object_set_sptr detections,
    const region_info& roi );

  vital::algo::image_object_detector_sptr m_detector;
  vital::logger_handle_t m_logger;
};


// =============================================================================
windowed_detector
::windowed_detector()
  : d( new priv() )
{
  attach_logger( "arrows.ocv.windowed_detector" );

  d->m_logger = logger();
}


windowed_detector
::~windowed_detector()
{}


// -----------------------------------------------------------------------------
vital::config_block_sptr
windowed_detector
::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "mode", d->m_mode,
    "Pre-processing resize option, can be: disabled, maintain_ar, scale, "
    "chip, chip_and_original, or adaptive." );
  config->set_value( "scale", d->m_scale,
    "Image scaling factor used when mode is scale or chip." );
  config->set_value( "chip_height", d->m_chip_height,
    "When in chip mode, the chip height." );
  config->set_value( "chip_width", d->m_chip_width,
    "When in chip mode, the chip width." );
  config->set_value( "chip_step_height", d->m_chip_step_height,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "chip_step_width", d->m_chip_step_width,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "chip_edge_filter", d->m_chip_edge_filter,
    "If using chipping, filter out detections this pixel count near borders." );
  config->set_value( "chip_adaptive_thresh", d->m_chip_adaptive_thresh,
    "If using adaptive selection, total pixel count at which we start to chip." );
  config->set_value( "batch_size", d->m_batch_size,
    "Optional processing batch size to send to the detector." );
  config->set_value( "min_detection_dim", d->m_min_detection_dim,
    "Minimum detection dimension in original image space." );
  config->set_value( "original_to_chip_size", d->m_original_to_chip_size,
    "Optionally enforce the input image is the specified chip size" );
  config->set_value( "black_pad", d->m_black_pad,
    "Black pad the edges of resized chips to ensure consistent dimensions" );

  vital::algo::image_object_detector::get_nested_algo_configuration(
    "detector", config, d->m_detector );

  return config;
}


// -----------------------------------------------------------------------------
void
windowed_detector
::set_configuration( vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing
  // a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  this->d->m_mode = config->get_value< std::string >( "mode" );
  this->d->m_scale = config->get_value< double >( "scale" );
  this->d->m_chip_width = config->get_value< int >( "chip_width" );
  this->d->m_chip_height = config->get_value< int >( "chip_height" );
  this->d->m_chip_step_width = config->get_value< int >( "chip_step_width" );
  this->d->m_chip_step_height = config->get_value< int >( "chip_step_height" );
  this->d->m_chip_edge_filter = config->get_value< int >( "chip_edge_filter" );
  this->d->m_chip_adaptive_thresh = config->get_value< int >( "chip_adaptive_thresh" );
  this->d->m_batch_size = config->get_value< int >( "batch_size" );
  this->d->m_min_detection_dim = config->get_value< int >( "min_detection_dim" );
  this->d->m_original_to_chip_size = config->get_value< bool >( "original_to_chip_size" );
  this->d->m_black_pad = config->get_value< bool >( "black_pad" );

  vital::algo::image_object_detector::set_nested_algo_configuration(
    "detector", config, d->m_detector );
}


// -----------------------------------------------------------------------------
bool
windowed_detector
::check_configuration( vital::config_block_sptr config ) const
{
  return vital::algo::image_object_detector::check_nested_algo_configuration(
    "detector", config );
}


// -----------------------------------------------------------------------------
vital::detected_object_set_sptr
windowed_detector
::detect( vital::image_container_sptr image_data ) const
{
  vital::scoped_wall_timer t( "Time to Detect Objects" );

  if( !image_data )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< vital::detected_object_set >();
  }

  cv::Mat cv_image = arrows::ocv::image_container::vital_to_ocv(
    image_data->get_image(), arrows::ocv::image_container::RGB_COLOR );

  std::string mode = d->m_mode;

  if( cv_image.rows == 0 || cv_image.cols == 0 )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< vital::detected_object_set >();
  }
  else if( mode == "adaptive" )
  {
    if( ( cv_image.rows * cv_image.cols ) >= d->m_chip_adaptive_thresh )
    {
      mode = "chip_and_original";
    }
    else if( d->m_original_to_chip_size )
    {
      mode = "maintain_ar";
    }
    else
    {
      mode = "disabled";
    }
  }

  cv::Mat cv_resized_image;

  vital::detected_object_set_sptr detections;

  // resizes image if enabled
  double scale_factor = 1.0;

  if( mode != "disabled" )
  {
    scale_factor = format_image( cv_image, cv_resized_image,
      ( mode == "original_and_resized" ? "scale" : mode ),
      d->m_scale, d->m_chip_width, d->m_chip_height );
  }
  else
  {
    cv_resized_image = cv_image;
  }

  // Run detector
  detections = std::make_shared< vital::detected_object_set >();

  cv::Rect original_dims( 0, 0, cv_image.cols, cv_image.rows );

  std::vector< cv::Mat > regions_to_process;
  std::vector< priv::region_info > region_properties;

  if( mode == "original_and_resized" )
  {
    cv::Mat scaled_original;

    if( cv_image.rows <= d->m_chip_height && cv_image.cols <= d->m_chip_width )
    {
      regions_to_process.push_back( cv_image );

      region_properties.push_back(
        priv::region_info( original_dims, 1.0 ) );
    }
    else
    {
      if( ( cv_image.rows * cv_image.cols ) >= d->m_chip_adaptive_thresh )
      {
        regions_to_process.push_back( cv_resized_image );

        region_properties.push_back(
          priv::region_info( original_dims, 1.0 / scale_factor ) );
      }

      double scaled_original_scale = scale_image_maintaining_ar( cv_image,
        scaled_original, d->m_chip_width, d->m_chip_height, d->m_black_pad );

      regions_to_process.push_back( scaled_original );

      region_properties.push_back(
        priv::region_info( original_dims, 1.0 / scaled_original_scale ) );
    }
  }
  else if( mode != "chip" && mode != "chip_and_original" )
  {
    regions_to_process.push_back( cv_resized_image );

    region_properties.push_back(
      priv::region_info( original_dims, 1.0 / scale_factor ) );
  }
  else
  {
    // Chip up scaled image
    for( int li = 0;
         li < cv_resized_image.cols - d->m_chip_width + d->m_chip_step_width;
         li += d->m_chip_step_width )
    {
      int ti = std::min( li + d->m_chip_width, cv_resized_image.cols );

      for( int lj = 0;
           lj < cv_resized_image.rows - d->m_chip_height + d->m_chip_step_height;
           lj += d->m_chip_step_height )
      {
        int tj = std::min( lj + d->m_chip_height, cv_resized_image.rows );

        if( tj-lj < 0 || ti-li < 0 )
        {
          continue;
        }

        cv::Rect resized_roi( li, lj, ti-li, tj-lj );
        cv::Rect original_roi( li / scale_factor,
                               lj / scale_factor,
                               (ti-li) / scale_factor,
                               (tj-lj) / scale_factor );

        cv::Mat cropped_chip = cv_resized_image( resized_roi );
        cv::Mat scaled_crop, tmp_cropped;

        double scaled_crop_scale = scale_image_maintaining_ar(
          cropped_chip, scaled_crop, d->m_chip_width, d->m_chip_height,
          d->m_black_pad );

        regions_to_process.push_back( scaled_crop );

        region_properties.push_back(
          priv::region_info( original_roi,
            d->m_chip_edge_filter,
            1.0 / scaled_crop_scale,
            li, lj,
            1.0 / scale_factor ) );
      }
    }

    // Extract full sized image chip if enabled
    if( mode == "chip_and_original" )
    {
      cv::Mat scaled_original;

      if( d->m_original_to_chip_size )
      {
        double scaled_original_scale = scale_image_maintaining_ar( cv_image,
          scaled_original, d->m_chip_width, d->m_chip_height, d->m_black_pad );

        regions_to_process.push_back( scaled_original );

        region_properties.push_back(
          priv::region_info( original_dims, 1.0 / scaled_original_scale ) );
      }
      else
      {
        regions_to_process.push_back( cv_image );

        region_properties.push_back(
          priv::region_info( original_dims, 1.0 ) );
      }
    }
  }

  // Process all regions
  unsigned max_count = d->m_batch_size;

  for( unsigned i = 0; i < regions_to_process.size(); i+= max_count )
  {
    unsigned batch_size = std::min( max_count,
      static_cast< unsigned >( regions_to_process.size() ) - i );

    std::vector< vital::image_container_sptr > imgs;

    for( unsigned j = 0; j < batch_size; j++ )
    {
      imgs.push_back(
        vital::image_container_sptr(
          new ocv::image_container( regions_to_process[i+j],
            ocv::image_container::RGB_COLOR ) ) );
    }

    std::vector< vital::detected_object_set_sptr > out =
      d->m_detector->batch_detect( imgs );

    for( unsigned j = 0; j < batch_size; j++ )
    {
      detections->add( d->scale_detections( out[ j ],
        region_properties[ i + j ] ) );
    }
  }

  const int min_dim = d->m_min_detection_dim;

  detections->filter([&min_dim](kwiver::vital::detected_object_sptr dos)
  {
    return !dos || dos->bounding_box().width() < min_dim
                || dos->bounding_box().height() < min_dim;
  });

  return detections;
} // windowed_detector::detect


vital::detected_object_set_sptr
windowed_detector::priv
::scale_detections(
  const vital::detected_object_set_sptr dets,
  const region_info& info )
{
  if( info.scale1 != 1.0 )
  {
    dets->scale( info.scale1 );
  }

  if( info.shiftx != 0 || info.shifty != 0 )
  {
    dets->shift( info.shiftx, info.shifty );
  }

  if( info.scale2 != 1.0 )
  {
    dets->scale( info.scale2 );
  }

  const int dist = info.edge_filter;

  if( dist <= 0 )
  {
    return dets;
  }

  const cv::Rect& roi = info.original_roi;

  std::vector< vital::detected_object_sptr > filtered_dets;

  for( auto det : *dets )
  {
    if( !det )
    {
      continue;
    }
    if( roi.x > 0 && det->bounding_box().min_x() < roi.x + dist )
    {
      continue;
    }
    if( roi.y > 0 && det->bounding_box().min_y() < roi.y + dist )
    {
      continue;
    }
    if( det->bounding_box().max_x() > roi.x + roi.width - dist )
    {
      continue;
    }
    if( det->bounding_box().max_y() > roi.y + roi.height - dist )
    {
      continue;
    }

    filtered_dets.push_back( det );
  }

  return vital::detected_object_set_sptr(
    new vital::detected_object_set( filtered_dets ) );
}


} } } // end namespace
