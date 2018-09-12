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

#include "ocv_stereo_depth_map.h"

#include <vital/vital_config.h>
#include <vital/types/image_container.h>
#include <vital/logger/logger.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <arrows/ocv/image_container.h>


namespace viame {

using namespace kwiver;

class ocv_stereo_depth_map::priv
{
public:

  int num_disparities;
  int sad_window_size;

#ifdef VIAME_OPENCV_VER_2
  cv::StereoBM algo;
#else
  cv::Ptr<cv::StereoBM> algo;
#endif

  priv()
    : num_disparities( 0 ),
      sad_window_size( 21 )
  {}

  ~priv()
  {}
};


ocv_stereo_depth_map::ocv_stereo_depth_map()
: d( new priv() )
{
}


ocv_stereo_depth_map::~ocv_stereo_depth_map()
{
}


// ---------------------------------------------------------------------------------------
vital::config_block_sptr
ocv_stereo_depth_map
::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "num_disparities", d->num_disparities, "Disparity count" );
  config->set_value( "sad_window_size", d->sad_window_size, "SAD window size" );

  return config;
}

// ---------------------------------------------------------------------------------------
void ocv_stereo_depth_map
::set_configuration( vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->num_disparities = config->get_value< int >( "num_disparities" );
  d->sad_window_size = config->get_value< int >( "sad_window_size" );

#ifdef VIAME_OPENCV_VER_2
  d->algo.init( d->num_disparities, d->sad_window_size );
#else
  d->algo = cv::StereoBM::create( d->num_disparities, d->sad_window_size );
#endif
}


// ---------------------------------------------------------------------------------------
bool ocv_stereo_depth_map
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}


// ---------------------------------------------------------------------------------------
kwiver::vital::image_container_sptr ocv_stereo_depth_map
::compute( kwiver::vital::image_container_sptr left_image,
           kwiver::vital::image_container_sptr right_image ) const
{
  cv::Mat ocv1 = arrows::ocv::image_container::vital_to_ocv( left_image->get_image() );
  cv::Mat ocv2 = arrows::ocv::image_container::vital_to_ocv( right_image->get_image() );

  cv::Mat output;

#ifdef VIAME_OPENCV_VER_2
  d->algo( ocv1, ocv2, output );
#else
  d->algo->compute( ocv1, ocv2, output );
#endif

  return kwiver::vital::image_container_sptr( new arrows::ocv::image_container( output ) );
}

} //end namespace viame
