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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

/**
 * \file
 * \brief Implementation of OCV refine detections draw debugging algorithm
 */

#include "refine_detections_draw.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <deque>

#include <vital/vital_config.h>
#include <vital/vital_foreach.h>
#include <vital/logger/logger.h>
#include <vital/exceptions/io.h>

#include <kwiversys/SystemTools.hxx>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

typedef kwiversys::SystemTools ST;

/// Private implementation class
class refine_detections_draw::priv
{
public:

  /// Constructor
  priv()
  : pattern( "detection_%10d.png" ),
    id( 0 )
  {
  }

  /// Destructor
  ~priv()
  {
  }

  /// Parameters
  std::string pattern;

  /// Variables
  unsigned id;
};


/// Constructor
refine_detections_draw
::refine_detections_draw()
: d_( new priv() )
{
}


/// Destructor
refine_detections_draw
::~refine_detections_draw()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
refine_detections_draw
::get_configuration() const
{
  vital::config_block_sptr config = vital::algo::refine_detections::get_configuration();

  config->set_value( "pattern", d_->pattern,
                     "The output pattern for writing images to disk." );

  return config;
}


/// Set this algorithm's properties via a config block
void
refine_detections_draw
::set_configuration( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d_->pattern = config->get_value<std::string>( "pattern" );
}

/// Check that the algorithm's currently configuration is valid
bool
refine_detections_draw
::check_configuration(vital::config_block_sptr config) const
{
  return true;
}

/// Output images with tracked features drawn on them
vital::detected_object_set_sptr
refine_detections_draw
::refine( vital::image_container_sptr image_data,
          vital::detected_object_set_sptr detections ) const
{
  cv::Mat img = ocv::image_container::vital_to_ocv( image_data->get_image() );

  VITAL_FOREACH( auto det, detections->select() )
  {
    // Generate output filename
    std::string ofn;
    size_t max_len = d_->pattern.size() + 4096;
    ofn.resize( max_len );
    int num_bytes = snprintf( &ofn[0], max_len, d_->pattern.c_str(), d_->id++ );

    if( num_bytes < 0 )
    {
      LOG_WARN( logger(), "Could not format output file name: \"" <<
        d_->pattern << "\". Disabling writing to disk." );
    }

    // Output image to file
    vital::bounding_box_d bbox = det->bounding_box();

    cv::Size s = img.size();
    vital::bounding_box_d bounds( vital::bounding_box_d::vector_type( 0, 0 ),
      vital::bounding_box_d::vector_type( s.width, s.height ) );

    bbox = intersection( bounds, bbox );

    // Make CV rect for out bbox coordinates
    cv::Rect r( bbox.upper_left()[0], bbox.upper_left()[1],
      bbox.width(), bbox.height() );

    cv::Mat crop = img( r );
    cv::imwrite( ofn, crop );
  }

  return detections;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
