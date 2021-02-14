/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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

#include "darknet_detector.h"
#include "darknet_custom_resize.h"

// kwiver includes
#include <vital/util/cpu_timer.h>
#include <vital/exceptions/io.h>
#include <vital/config/config_block_formatter.h>

#include <arrows/ocv/image_container.h>
#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <exception>
#include <limits>

#include "darknet/yolo_v2_class.hpp"

namespace kwiver {
namespace arrows {
namespace darknet {

// =============================================================================
class darknet_detector::priv
{
public:
  priv()
    : m_thresh( 0.24 )
    , m_hier_thresh( 0.5 )
    , m_gpu_index( -1 )
    , m_resize_option( "disabled" )
    , m_scale( 1.0 )
    , m_chip_step( 100 )
    , m_nms_threshold( 0.4 )
    , m_gs_to_rgb( true )
    , m_chip_edge_filter( 0 )
    , m_chip_adaptive_thresh( 2000000 )
    , m_is_first( true )
    , m_names()
  {}

  ~priv() {}

  // Items from the config
  std::string m_net_config;
  std::string m_weight_file;
  std::string m_class_names;

  float m_thresh;
  float m_hier_thresh;
  int m_gpu_index;

  std::string m_resize_option;
  double m_scale;
  int m_chip_step;
  double m_nms_threshold;
  bool m_gs_to_rgb;
  int m_chip_edge_filter;
  int m_chip_adaptive_thresh;
  bool m_is_first;

  // Needed to operate the model
  std::vector< std::string > m_names;
  std::unique_ptr< Detector > m_net;

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

  std::vector< vital::detected_object_set_sptr > process_images(
    const std::vector< cv::Mat >& cv_image );

  vital::detected_object_set_sptr scale_detections(
    const vital::detected_object_set_sptr detections,
    const region_info& roi );

  kwiver::vital::logger_handle_t m_logger;
};


// =============================================================================
darknet_detector
::darknet_detector()
  : d( new priv() )
{
  attach_logger( "arrows.darknet.darknet_detector" );
  d->m_logger = logger();
}


darknet_detector
::~darknet_detector()
{}


// -----------------------------------------------------------------------------
vital::config_block_sptr
darknet_detector
::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "net_config", d->m_net_config,
    "Name of network config file." );
  config->set_value( "weight_file", d->m_weight_file,
    "Name of optional weight file." );
  config->set_value( "class_names", d->m_class_names,
    "Name of file that contains the class names." );
  config->set_value( "thresh", d->m_thresh,
    "Threshold value." );
  config->set_value( "hier_thresh", d->m_hier_thresh,
    "Hier threshold value." );
  config->set_value( "gpu_index", d->m_gpu_index,
    "GPU index. Only used when darknet is compiled with GPU support." );
  config->set_value( "resize_option", d->m_resize_option,
    "Pre-processing resize option, can be: disabled, maintain_ar, scale, "
    "chip, chip_and_original, or adaptive." );
  config->set_value( "scale", d->m_scale,
    "Image scaling factor used when resize_option is scale or chip." );
  config->set_value( "chip_step", d->m_chip_step,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "nms_threshold", d->m_nms_threshold,
    "Non-maximum suppression threshold." );
  config->set_value( "gs_to_rgb", d->m_gs_to_rgb,
    "Convert input greyscale images to rgb before processing." );
  config->set_value( "chip_edge_filter", d->m_chip_edge_filter,
    "If using chipping, filter out detections this pixel count near borders." );
  config->set_value( "chip_adaptive_thresh", d->m_chip_adaptive_thresh,
    "If using adaptive selection, total pixel count at which we start to chip." );

  return config;
}


// -----------------------------------------------------------------------------
void
darknet_detector
::set_configuration( vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing
  // a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  d->m_net_config  = config->get_value< std::string >( "net_config" );
  d->m_weight_file = config->get_value< std::string >( "weight_file" );
  d->m_class_names = config->get_value< std::string >( "class_names" );
  d->m_thresh      = config->get_value< float >( "thresh" );
  d->m_hier_thresh = config->get_value< float >( "hier_thresh" );
  d->m_gpu_index   = config->get_value< int >( "gpu_index" );
  d->m_resize_option = config->get_value< std::string >( "resize_option" );
  d->m_scale       = config->get_value< double >( "scale" );
  d->m_chip_step   = config->get_value< int >( "chip_step" );
  d->m_nms_threshold = config->get_value< double >( "nms_threshold" );
  d->m_gs_to_rgb   = config->get_value< bool >( "gs_to_rgb" );
  d->m_chip_edge_filter = config->get_value< int >( "chip_edge_filter" );
  d->m_chip_adaptive_thresh = config->get_value< int >( "chip_adaptive_thresh" );

  // TODO Open file and return 'list' of labels
  //d->m_names = 

  d->m_net.reset( new Detector( d->m_net_config, d->m_weight_file, d->m_gpu_index ) );

  // This assumes that there are no other users of random number
  // generator in this application.
  srand( 2222222 );
} // darknet_detector::set_configuration


// -----------------------------------------------------------------------------
bool
darknet_detector
::check_configuration( vital::config_block_sptr config ) const
{
  std::string net_config = config->get_value<std::string>( "net_config" );
  std::string class_file = config->get_value<std::string>( "class_names" );

  bool success = true;

  if( net_config.empty() )
  {
    std::stringstream str;
    kwiver::vital::config_block_formatter fmt( config );
    fmt.print( str );
    LOG_ERROR( logger(), "Required net config file not specified. "
      "Configuration is as follows:\n" << str.str() );
    success = false;
  }
  else if( !kwiversys::SystemTools::FileExists( net_config ) )
  {
    LOG_ERROR( logger(), "net config file \"" << net_config << "\" not found." );
    success = false;
  }

  if( class_file.empty() )
  {
    std::stringstream str;
    kwiver::vital::config_block_formatter fmt( config );
    fmt.print( str );
    LOG_ERROR( logger(), "Required class name list file not specified, "
      "Configuration is as follows:\n" << str.str() );
    success = false;
  }
  else if( ! kwiversys::SystemTools::FileExists( class_file ) )
  {
    LOG_ERROR( logger(), "class names file \"" << class_file << "\" not found." );
    success = false;
  }

  return success;
} // darknet_detector::check_configuration


// -----------------------------------------------------------------------------
vital::detected_object_set_sptr
darknet_detector
::detect( vital::image_container_sptr image_data ) const
{
  kwiver::vital::scoped_cpu_timer t( "Time to Detect Objects" );

  if( !image_data )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< vital::detected_object_set >();
  }

  cv::Mat cv_image = kwiver::arrows::ocv::image_container::vital_to_ocv(
    image_data->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR );

  if( cv_image.rows == 0 || cv_image.cols == 0 )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< vital::detected_object_set >();
  }
  else if( d->m_resize_option == "adaptive" )
  {
    if( ( cv_image.rows * cv_image.cols ) >= d->m_chip_adaptive_thresh )
    {
      d->m_resize_option = "chip_and_original";
    }
    else
    {
      d->m_resize_option = "maintain_ar";
    }
  }

  cv::Mat cv_resized_image;

  vital::detected_object_set_sptr detections;

  // resizes image if enabled
  double scale_factor = 1.0;

  if( d->m_resize_option != "disabled" )
  {
    scale_factor = format_image( cv_image, cv_resized_image,
      d->m_resize_option, d->m_scale,
      d->m_net->get_net_width(), d->m_net->get_net_height() );
  }
  else
  {
    cv_resized_image = cv_image;
  }

  if( d->m_gs_to_rgb && cv_resized_image.channels() == 1 )
  {
    cv::Mat color_image;
    cv::cvtColor( cv_resized_image, color_image, CV_GRAY2RGB );
    cv_resized_image = color_image;
  }

  // Run detector
  detections = std::make_shared< vital::detected_object_set >();

  cv::Rect original_dims( 0, 0, cv_image.cols, cv_image.rows );

  std::vector< cv::Mat > regions_to_process;
  std::vector< priv::region_info > region_properties;

  if( d->m_resize_option != "chip" && d->m_resize_option != "chip_and_original" )
  {
    regions_to_process.push_back( cv_resized_image );

    region_properties.push_back(
      priv::region_info( original_dims, 1.0 / scale_factor ) );
  }
  else
  {
    // Chip up scaled image
    for( int li = 0;
         li < cv_resized_image.cols - d->m_net->get_net_width() + d->m_chip_step;
         li += d->m_chip_step )
    {
      int ti = std::min( li + d->m_net->get_net_width(), cv_resized_image.cols );

      for( int lj = 0;
           lj < cv_resized_image.rows - d->m_net->get_net_height() + d->m_chip_step;
           lj += d->m_chip_step )
      {
        int tj = std::min( lj + d->m_net->get_net_height(), cv_resized_image.rows );

        cv::Rect resized_roi( li, lj, ti-li, tj-lj );
        cv::Rect original_roi( li / scale_factor,
                               lj / scale_factor,
                               (ti-li) / scale_factor,
                               (tj-lj) / scale_factor );

        cv::Mat cropped_chip = cv_resized_image( resized_roi );
        cv::Mat scaled_crop, tmp_cropped;

        double scaled_crop_scale = scale_image_maintaining_ar(
          cropped_chip, scaled_crop, d->m_net->get_net_width(), d->m_net->get_net_height() );

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
    if( d->m_resize_option == "chip_and_original" )
    {
      cv::Mat scaled_original;

      double scaled_original_scale = scale_image_maintaining_ar( cv_image,
        scaled_original, d->m_net->get_net_width(), d->m_net->get_net_height() );

      if( d->m_gs_to_rgb && scaled_original.channels() == 1 )
      {
        cv::Mat color_image;
        cv::cvtColor( scaled_original, color_image, CV_GRAY2RGB );
        scaled_original = color_image;
      }

      regions_to_process.push_back( scaled_original );

      region_properties.push_back(
        priv::region_info( original_dims, 1.0 / scaled_original_scale ) );
    }
  }

  // Process all regions
  unsigned max_count = 1;

  for( unsigned i = 0; i < regions_to_process.size(); i+= max_count )
  {
    unsigned batch_size = std::min( max_count,
      static_cast< unsigned >( regions_to_process.size() ) - i );

    std::vector< cv::Mat > imgs;

    for( unsigned j = 0; j < batch_size; j++ )
    {
      imgs.push_back( regions_to_process[ i + j ] );
    }

    std::vector< vital::detected_object_set_sptr > out = d->process_images( imgs );

    for( unsigned j = 0; j < batch_size; j++ )
    {
      detections->add( d->scale_detections( out[ j ], region_properties[ i + j ] ) );
    }
  }

  return detections;
} // darknet_detector::detect


// =============================================================================
std::vector< vital::detected_object_set_sptr >
darknet_detector::priv
::process_images( const std::vector< cv::Mat >& cv_images )
{
  std::vector< vital::detected_object_set_sptr > output;

  for( unsigned i = 0; i < cv_images.size(); i++ )
  {
    auto darknet_output = m_net->detect( cv_images[i], m_thresh );
    auto detected_objects = std::make_shared< vital::detected_object_set >();

    for( const auto& det : darknet_output )
    {
      kwiver::vital::bounding_box_d bbox( det.x, det.y, det.x + det.w, det.y + det.h );
      auto dot = std::make_shared< kwiver::vital::detected_object_type >(
        m_names[ det.obj_id ], det.prob );

      detected_objects->add(
        std::make_shared< kwiver::vital::detected_object >(
          bbox, det.prob, dot ) );
    }

    output.push_back( detected_objects );
  }

  return output;
}


vital::detected_object_set_sptr
darknet_detector::priv
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
