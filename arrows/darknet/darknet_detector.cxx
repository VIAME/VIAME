/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
#include <vital/logger/logger.h>
#include <vital/util/cpu_timer.h>
#include <vital/vital_foreach.h>

#include <arrows/ocv/image_container.h>
#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <sstream>
#include <exception>

#ifdef DARKNET_USE_GPU
#define GPU
#include <cuda_runtime.h>
#endif

// darknet includes
extern "C" {

#include "cuda.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "image.h"

}

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
    , m_resize_i( 0 )
    , m_resize_j( 0 )
    , m_chip_step( 100 )
    , m_gs_to_rgb( true )
    , m_names( 0 )
    , m_boxes( 0 )
    , m_probs( 0 )
  { }

  ~priv()
  {
    free( m_names );
  }

  // Items from the config
  std::string m_net_config;
  std::string m_weight_file;
  std::string m_class_names;

  float m_thresh;
  float m_hier_thresh;
  int m_gpu_index;

  std::string m_resize_option;
  double m_scale;
  int m_resize_i;
  int m_resize_j;
  int m_chip_step;
  bool m_gs_to_rgb;

  // Needed to operate the model
  char **m_names;                 /* list of classes/labels */
  network m_net;

  box *m_boxes;                   /* detection boxes */
  float **m_probs;                /*  */

  // Helper functions
  image cvmat_to_image( const cv::Mat& src );
  vital::detected_object_set_sptr process_image( const cv::Mat& cv_image );

  kwiver::vital::logger_handle_t m_logger;
};


// =============================================================================
darknet_detector::
darknet_detector()
  : d( new priv() )
{
  // set darknet global GPU index
  gpu_index = d->m_gpu_index;
}


darknet_detector::
~darknet_detector()
{}


// -----------------------------------------------------------------------------
vital::config_block_sptr
darknet_detector::
get_configuration() const
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
    "chip, or chip_and_original." );
  config->set_value( "scale", d->m_scale,
    "Image scaling factor used when resize_option is scale or chip." );
  config->set_value( "resize_ni", d->m_resize_i,
    "Width resolution after resizing" );
  config->set_value( "resize_nj", d->m_resize_j,
    "Height resolution after resizing" );
  config->set_value( "chip_step", d->m_chip_step,
    "When in chip mode, the chip step size between chips." );
  config->set_value( "gs_to_rgb", d->m_gs_to_rgb,
    "Convert input greyscale images to rgb before processing." );

  return config;
}


// -----------------------------------------------------------------------------
void
darknet_detector::
set_configuration( vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing
  // a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  this->d->m_net_config  = config->get_value< std::string >( "net_config" );
  this->d->m_weight_file = config->get_value< std::string >( "weight_file" );
  this->d->m_class_names = config->get_value< std::string >( "class_names" );
  this->d->m_thresh      = config->get_value< float >( "thresh" );
  this->d->m_hier_thresh = config->get_value< float >( "hier_thresh" );
  this->d->m_gpu_index   = config->get_value< int >( "gpu_index" );
  this->d->m_resize_option = config->get_value< std::string >( "resize_option" );
  this->d->m_scale       = config->get_value< double >( "scale" );
  this->d->m_resize_i    = config->get_value< int >( "resize_ni" );
  this->d->m_resize_j    = config->get_value< int >( "resize_nj" );
  this->d->m_chip_step   = config->get_value< int >( "chip_step" );
  this->d->m_gs_to_rgb   = config->get_value< bool >( "gs_to_rgb" );

  /* the size of this array is a mystery - probably has to match some
   * constant in net description */

#ifdef DARKNET_USE_GPU
  if( d->m_gpu_index >= 0 )
  {
    cuda_set_device( d->m_gpu_index );
  }
#endif

  // Open file and return 'list' of labels
  d->m_names = get_labels( const_cast< char* >( d->m_class_names.c_str() ) );

  d->m_net = parse_network_cfg( const_cast< char* >( d->m_net_config.c_str() ) );
  if( ! d->m_weight_file.empty() )
  {
    load_weights( &d->m_net, const_cast< char* >( d->m_weight_file.c_str() ) );
  }

  set_batch_network( &d->m_net, 1 );

  // This assumes that there are no other users of random number
  // generator in this application.
  srand( 2222222 );
} // darknet_detector::set_configuration


// -----------------------------------------------------------------------------
bool
darknet_detector::
check_configuration( vital::config_block_sptr config ) const
{
  std::string net_config = config->get_value<std::string>( "net_config" );
  std::string class_file = config->get_value<std::string>( "class_names" );

  bool success( true );

  if( net_config.empty() )
  {
    std::stringstream str;
    config->print( str );
    LOG_ERROR( logger(), "Required net config file not specified. "
      "Configuration is as follows:\n" << str.str() );
    success = false;
  }
  else if( ! kwiversys::SystemTools::FileExists( net_config ) )
  {
    LOG_ERROR( logger(), "net config file \"" << net_config << "\" not found." );
    success = false;
  }

  if( class_file.empty() )
  {
    std::stringstream str;
    config->print( str );
    LOG_ERROR( logger(), "Required class name list file not specified, "
      "Configuration is as follows:\n" << str.str() );
    success = false;
  }
  else if( ! kwiversys::SystemTools::FileExists( class_file ) )
  {
    LOG_ERROR( logger(), "class names file \"" << class_file << "\" not found." );
    success = false;
  }

  if( d->m_resize_option != "disabled" &&
      ( d->m_resize_i != 0 || d->m_resize_j != 0 || d->m_resize_option == "scale" ) )
  {
    LOG_ERROR( logger(), "resize dimentions must be set if resizing enabled" );
    success = false;
  }

  return success;
} // darknet_detector::check_configuration


// -----------------------------------------------------------------------------
vital::detected_object_set_sptr
darknet_detector::
detect( vital::image_container_sptr image_data ) const
{
  kwiver::vital::scoped_cpu_timer t( "Time to Detect Objects" );

  cv::Mat cv_image =
    kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );

  cv::Mat cv_resized_image;

  vital::detected_object_set_sptr detections;

  // resizes image if enabled
  double scale_factor = 1.0;

  if( d->m_resize_option != "disabled" )
  {
    scale_factor = format_image( cv_image, cv_resized_image,
      d->m_resize_option, d->m_scale, d->m_resize_i, d->m_resize_j );
  }
  else
  {
    cv_resized_image = cv_image;
  }

  if( d->m_gs_to_rgb && cv_resized_image.channels() == 1 )
  {
    cv::Mat color_image;
    cv::cvtColor( cv_resized_image, color_image, CV_GRAY2BGR );
    cv_resized_image = color_image;
  }

  // run detector
  if( d->m_resize_option != "chip" && d->m_resize_option != "chip_and_original" )
  {
    detections = d->process_image( cv_resized_image );

    // rescales output detections if required
    detections->scale( 1.0 / scale_factor );
  }
  else
  {
    detections = std::make_shared< vital::detected_object_set >();

    // Chip up and process scaled image
    for( int li = 0; li < cv_resized_image.cols; li += d->m_chip_step )
    {
      int ti = std::min( li + d->m_resize_i, cv_resized_image.cols );

      for( int lj = 0; lj < cv_resized_image.rows; lj += d->m_chip_step )
      {
        int tj = std::min( lj + d->m_resize_j, cv_resized_image.rows );

        cv::Mat cropped_image = cv_resized_image( cv::Rect( li, lj, ti-li, tj-lj ) );
        cv::Mat scaled_crop, tmp_cropped;

        double scaled_crop_scale = scale_image_maintaining_ar(
          cropped_image, scaled_crop, d->m_resize_i, d->m_resize_j );
        cv::cvtColor(scaled_crop, tmp_cropped, cv::COLOR_BGR2RGB);

        vital::detected_object_set_sptr new_dets = d->process_image( tmp_cropped );
        new_dets->scale( 1.0 / scaled_crop_scale );
        new_dets->shift( li, lj );
        new_dets->scale( 1.0 / scale_factor );


        detections->add( new_dets );
      }
    }

    // Process full sized image if enabled
    if( d->m_resize_option == "chip_and_original" )
    {
      cv::Mat scaled_original;

      double scaled_original_scale = scale_image_maintaining_ar( cv_image,
        scaled_original, d->m_resize_i, d->m_resize_j );

      vital::detected_object_set_sptr new_dets = d->process_image( scaled_original );

      new_dets->scale( 1.0 / scaled_original_scale );

      detections->add( new_dets );
    }
  }

  return detections;
} // darknet_detector::detect


// =============================================================================
vital::detected_object_set_sptr
darknet_detector::priv::
process_image( const cv::Mat& cv_image )
{
  // copies and converts to floating pixel value.
  image im = cvmat_to_image( cv_image );
  // show_image( im, "first version" );

  image sized = resize_image( im, m_net.w, m_net.h );
  // show_image( sized, "sized version" );

  layer l = m_net.layers[m_net.n - 1];     /* last network layer (output?) */
  const size_t l_size = l.w * l.h * l.n;

  m_boxes = (box*) calloc( l_size, sizeof( box ) );
  m_probs = (float**) calloc( l_size, sizeof( float* ) ); // allocate vector of pointers
  for( size_t j = 0; j < l_size; ++j )
  {
    m_probs[j] = (float*) calloc( l.classes + 1, sizeof( float*) );
  }

  /* pointer the image data */
  float* X = sized.data;

  /* run image through network */
  network_predict( m_net, X );

  /* get boxes around detected objects */
  get_region_boxes( l,        /* i: network output layer */
                    1, 1,     /* i: w, h -  */
                    m_thresh, /* i: caller supplied threshold */
                    m_probs,  /* o: probability vector */
                    m_boxes,  /* o: list of boxes */
                    0,        /* i: only objectness (false) */
                    0,        /* i: map */
                    m_hier_thresh ); /* i: caller supplied value */

  const float nms( 0.4 );       // don't know what this is

  if( l.softmax_tree && nms )
  {
    do_nms_obj( m_boxes, m_probs, l_size, l.classes, nms );
  }
  else if( nms )
  {
    do_nms_sort( m_boxes, m_probs, l_size, l.classes, nms );
  }
  else
  {
    LOG_ERROR( m_logger, "Internal error - nms == 0" );
  }

  // -- extract detections and convert to our format --
  auto detected_objects = std::make_shared< vital::detected_object_set >();

  for( size_t i = 0; i < l_size; ++i )
  {
    const box b = m_boxes[i];

    int left  = ( b.x - b.w / 2. ) * im.w;
    int right = ( b.x + b.w / 2. ) * im.w;
    int top   = ( b.y - b.h / 2. ) * im.h;
    int bot   = ( b.y + b.h / 2. ) * im.h;

    /* clip box to image bounds */
    if( left < 0 )
    {
      left = 0;
    }
    if( right > im.w - 1 )
    {
      right = im.w - 1;
    }
    if( top < 0 )
    {
      top = 0;
    }
    if( bot > im.h - 1 )
    {
      bot = im.h - 1;
    }

    kwiver::vital::bounding_box_d bbox( left, top, right, bot );

    auto dot = std::make_shared< kwiver::vital::detected_object_type >();
    bool has_name = false;

    // Iterate over all classes and collect all names over the threshold, and max score
    double conf = 0.0;

    for( int class_idx = 0; class_idx < l.classes; ++class_idx )
    {
      const double prob = static_cast< double >( m_probs[i][class_idx] );

      if( prob >= m_thresh )
      {
        const std::string class_name( m_names[class_idx] );
        dot->set_score( class_name, prob );
        conf = std::max( conf, prob );
        has_name = true;
      }
    }

    if( has_name )
    {
      detected_objects->add(
        std::make_shared< kwiver::vital::detected_object >( bbox, conf, dot ) );
    }
  }

  // Free allocated memory
  free_image(im);
  free_image(sized);
  free( m_boxes );
  free_ptrs( (void**)m_probs, l_size );

  return detected_objects;
}


image
darknet_detector::priv::
cvmat_to_image( const cv::Mat& src )
{
  // accept only char type matrices
  CV_Assert( src.depth() == CV_8U );

  unsigned char *data = (unsigned char *)src.data;
  int h = src.rows; // src.height;
  int w = src.cols; // src.width;
  int c = src.channels(); // src.nChannels;
  int step = w * c; // src.widthStep;
  image out = make_image(w, h, c);
  int i, j, k, count=0;;

  for( k = c-1; k >= 0 ; --k )
  {
    for( i = 0; i < h; ++i )
    {
      for( j = 0; j < w; ++j )
      {
        out.data[count++] = data[i*step + j*c + k]/255.;
      }
    }
  }

  return out;
}

} } } // end namespace
