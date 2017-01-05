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

// kwiver includes
#include <vital/logger/logger.h>
#include <vital/util/cpu_timer.h>
#include <arrows/ocv/image_container.h>
#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>

#include <string>


// darknet includes
extern "C" {

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

// ==================================================================
class darknet_detector::priv
{
public:
  priv()
    : m_thresh(.24)
    , m_hier_thresh(5)
    , m_names( 0 )
    , m_boxes( 0 )
    , m_probs( 0 )
  { }

  // copy CTOR
  priv(const priv& other)
    : m_data_config( other.m_data_config )
    , m_net_config( other.m_net_config )
    , m_weight_file( other.m_weight_file )
    , m_thresh( other.m_thresh )
    , m_hier_thresh( other.m_hier_thresh )
      //+ other stuff
    , m_logger( kwiver::vital::get_logger( "arrows.darknet.darknet_detector" ) )
  { }

  ~priv()
  {
    free(m_names);
  }

  image cvmat_to_image( const cv::Mat& src );


  // Items from the config
  std::string m_data_config;
  std::string m_net_config;
  std::string m_weight_file;

  float m_thresh;
  float m_hier_thresh;

  // Needed to operate the model
  char **m_names;                 /* list of classes/labels */
  network m_net;

  layer m_l;                      /* output layer of network */
  box *m_boxes;                   /* detection boxes */
  float **m_probs;                /*  */

  kwiver::vital::logger_handle_t m_logger;
};


// ==================================================================
darknet_detector::
darknet_detector()
  : d( new priv() )
{ }


darknet_detector::
darknet_detector( darknet_detector const& frd )
  : d( new priv( *frd.d ) )
{ }


darknet_detector::
~darknet_detector()
{ }


// --------------------------------------------------------------------
vital::config_block_sptr
darknet_detector::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "data_config", d->m_data_config, "Name of data config file." );
  config->set_value( "net_config", d->m_net_config, "Name of network config file." );
  config->set_value( "weight_file", d->m_weight_file, "Name of optional weight file." );
  config->set_value( "thresh", d->m_thresh, "Threshold value." );
  config->set_value( "hier_thresh", d->m_hier_thresh, "Hier threshold value." );

  return config;
}


// --------------------------------------------------------------------
void
darknet_detector::
set_configuration( vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  this->d->m_data_config = config->get_value< std::string > ( "data_config" );
  this->d->m_net_config  = config->get_value< std::string > ( "net_config" );
  this->d->m_weight_file = config->get_value< std::string > ( "weight_file" );
  this->d->m_thresh      = config->get_value< float > ( "thresh" );
  this->d->m_hier_thresh = config->get_value< float > ( "hier_thresh" );

  /* Read configuration file */
  list* options = read_data_cfg( const_cast< char* >(d->m_data_config.c_str()) );

  /* find optional specification for class names */
  //+ This could become an additional parameter for the name of the class names
  char* name_list = option_find_str( options, "names", "data/names.list" );

  /* the size of this array is a mystery - probably has to match some
   * constant in net description */

  // Open file and return 'list' of labels
  d->m_names = get_labels( name_list );

  d->m_net = parse_network_cfg( const_cast< char* >(d->m_net_config.c_str()) );
  if ( d->m_weight_file.empty() )
  {
    load_weights( &d->m_net, const_cast< char* >(d->m_weight_file.c_str()) );
  }

  set_batch_network( &d->m_net, 1 );

  // This assumes that there are no other users of random number
  // generator in this application.
  srand( 2222222 );

  // Clean up extra storage
  free_list( options );

} // darknet_detector::set_configuration


// --------------------------------------------------------------------
bool
darknet_detector::
check_configuration( vital::config_block_sptr config ) const
{

  std::string data_config = config->get_value<std::string>( "data_config" );
  std::string net_config = config->get_value<std::string>( "net_config" );

  bool success( true );

  if ( data_config.empty() )
  {
    LOG_ERROR( d->m_logger, "Required data config file not specified" );
    success = false;
  }
  else if ( ! kwiversys::SystemTools::FileExists( data_config ) )
  {
    LOG_ERROR( d->m_logger, "data config file \"" << data_config << "\" not found." );
    success = false;
  }

  if ( net_config.empty() )
  {
    LOG_ERROR( d->m_logger, "Required net config file not specified" );
    success = false;
  }
  else if ( ! kwiversys::SystemTools::FileExists( net_config ) )
  {
    LOG_ERROR( d->m_logger, "net config file \"" << net_config << "\" not found." );
    success = false;
  }

  return success;
} // darknet_detector::check_configuration


// --------------------------------------------------------------------
vital::detected_object_set_sptr
darknet_detector::
detect( vital::image_container_sptr image_data ) const
{
  kwiver::vital::scoped_cpu_timer t( "Time to Detect Objects" );
  cv::Mat cv_image = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );

  // copies and converts to floating pixel value.
  image im = d->cvmat_to_image( cv_image );

  image sized = resize_image( im, d->m_net.w, d->m_net.h );
  d->m_l = d->m_net.layers[d->m_net.n - 1];     /* last network layer (output?) */

  const size_t l_size = d->m_l.w * d->m_l.h * d->m_l.n;

  //+ do these need to be cleared each time?
  d->m_boxes = (box*) calloc( l_size, sizeof( box ) );
  d->m_probs = (float**) calloc( l_size, sizeof( float* ) ); // allocate vector of pointers

  for ( size_t j = 0; j < l_size; ++j )
  {
    d->m_probs[j] = (float*) calloc( d->m_l.classes + 1, sizeof( float*) );
  }
  //+ end of allocation question

  /* pointer te image data */
  float* X = sized.data;

  /* run image through network */
  network_predict( d->m_net, X );

  /* get boxes around detected objects */
  get_region_boxes( d->m_l,     /* i: network output layer */
                    1, 1, /* i: w, h -  */
                    d->m_thresh, /* i: caller supplied threshold */
                    d->m_probs, /* o: probability vector */
                    d->m_boxes, /* o: list of boxes */
                    0,     /* i: only objectness (false) */
                    0,     /* i: map */
                    d->m_hier_thresh ); /* i: caller supplied value */

  const float nms( 0.4 );       // don't know what this is

  if ( d->m_l.softmax_tree && nms )
  {
    do_nms_obj( d->m_boxes, d->m_probs, l_size, d->m_l.classes, nms );
  }
  else if ( nms )
  {
    do_nms_sort( d->m_boxes, d->m_probs, l_size, d->m_l.classes, nms );
  }
  else
  {
    LOG_ERROR( d->m_logger, "Internal error - nms == 0" );
  }

  // -- extract detections and convert to our format --
  auto detected_objects = std::make_shared< vital::detected_object_set > ();

  for ( size_t i = 0; i < l_size; ++i )
  {
    //+ there is a way to get more than one class name from prob matrix.
    //+ want all classes above threshold
    int class_idx = max_index( d->m_probs[i], d->m_l.classes );
    const float prob = d->m_probs[i][class_idx];

    if ( prob > d->m_thresh )
    {
      std::string class_name( d->m_names[class_idx] );

      const box b = d->m_boxes[i];

      int left  = ( b.x - b.w / 2. ) * im.w;
      int right = ( b.x + b.w / 2. ) * im.w;
      int top   = ( b.y - b.h / 2. ) * im.h;
      int bot   = ( b.y + b.h / 2. ) * im.h;

      /* clip box to image bounds */
      if ( left < 0 )
      {
        left = 0;
      }
      if ( right > im.w - 1 )
      {
        right = im.w - 1;
      }
      if ( top < 0 )
      {
        top = 0;
      }
      if ( bot > im.h - 1 )
      {
        bot = im.h - 1;
      }

      kwiver::vital::bounding_box_d bbox( left, top, right, bot);

      auto dot = std::make_shared< kwiver::vital::detected_object_type >();
      dot->set_score( class_name, prob );

      detected_objects->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
    } // end for loop
  }

  // Free allocated memory
  free( d->m_boxes );
  free_ptrs( (void**)d->m_probs, l_size );

  return detected_objects;
} // darknet_detector::detect


// ==================================================================

image
darknet_detector::priv::
cvmat_to_image( const cv::Mat& src )
{
  // accept only char type matrices
  CV_Assert(src.depth() == CV_8U);

  const int channels = src.channels();
  int nRows = src.rows;
  int nCols = src.cols * channels;

  image out = make_image( nCols, nRows, channels );

  if (src.isContinuous())
  {
    nCols *= nRows;
    nRows = 1;
  }

  int count = 0;
  for( int i = 0; i < nRows; ++i)
  {
    uchar* p = const_cast< uchar* >( src.ptr<uchar>(i) );
    for ( int j = 0; j < nCols; ++j)
    {
      out.data[count++] = p[j]/255.0;
    }
  }

  return out;
}


} } } // end namespace
