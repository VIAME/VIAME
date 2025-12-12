/*ckwg +29
 * Copyright 2017-2025 by Kitware, Inc.
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
#include "ocv_stereo_calibration.h"

#include <vital/vital_config.h>
#include <vital/types/image_container.h>
#include <vital/logger/logger.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <arrows/ocv/image_container.h>

namespace kv = kwiver::vital;

namespace viame {

class ocv_stereo_depth_map::priv
{
public:

  std::string algorithm{ "BM" };
  int min_disparity{ 0 };
  int num_disparities{ 16 };
  int sad_window_size{ 21 };
  int block_size{ 3 };
  int speckle_window_size{ 50 };
  int speckle_range{ 5 };

  cv::Ptr<cv::StereoMatcher> algo;
};


ocv_stereo_depth_map::ocv_stereo_depth_map()
: d( new priv() )
{
}


ocv_stereo_depth_map::~ocv_stereo_depth_map()
{
}


// ---------------------------------------------------------------------------------------
kv::config_block_sptr
ocv_stereo_depth_map
::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "algorithm", d->algorithm, "Algorithm: BM or SGBM" );
  config->set_value( "min_disparity", d->min_disparity, "Min Disparity" );
  config->set_value( "num_disparities", d->num_disparities, "Disparity count" );
  config->set_value( "sad_window_size", d->sad_window_size, "SAD window size" );
  config->set_value( "block_size", d->block_size, "Block size" );
  config->set_value( "speckle_window_size", d->speckle_window_size, "Speckle Window Size" );
  config->set_value( "speckle_range", d->speckle_range, "Speckle Range" );

  return config;
}

// ---------------------------------------------------------------------------------------
void ocv_stereo_depth_map
::set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->algorithm = config->get_value< std::string >( "algorithm" );
  d->min_disparity = config->get_value< int >( "min_disparity" );
  d->num_disparities = config->get_value< int >( "num_disparities" );
  d->sad_window_size = config->get_value< int >( "sad_window_size" );
  d->block_size = config->get_value< int >( "block_size" );
  d->speckle_window_size = config->get_value< int >( "speckle_window_size" );
  d->speckle_range = config->get_value< int >( "speckle_range" );

  if( d->algorithm == "BM" )
  {
    d->algo = cv::StereoBM::create( d->num_disparities, d->sad_window_size );
    d->algo->setSpeckleWindowSize( d->speckle_window_size );
    d->algo->setSpeckleRange( d->speckle_range );
  }
  else if( d->algorithm == "SGBM" )
  {
    d->algo = cv::StereoSGBM::create( d->min_disparity, d->num_disparities, d->block_size );
    d->algo->setSpeckleWindowSize( d->speckle_window_size );
    d->algo->setSpeckleRange( d->speckle_range );
  }
  else
  {
    throw std::runtime_error( "Invalid algorithm type " + d->algorithm );
  }
}


// ---------------------------------------------------------------------------------------
bool ocv_stereo_depth_map
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}


// ---------------------------------------------------------------------------------------
kv::image_container_sptr ocv_stereo_depth_map
::compute( kv::image_container_sptr left_image,
           kv::image_container_sptr right_image ) const
{
  cv::Mat ocv1 = kwiver::arrows::ocv::image_container::vital_to_ocv(
    left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat ocv2 = kwiver::arrows::ocv::image_container::vital_to_ocv(
    right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Convert to grayscale using shared utility
  cv::Mat ocv1_gray = stereo_calibration::to_grayscale( ocv1 );
  cv::Mat ocv2_gray = stereo_calibration::to_grayscale( ocv2 );

  cv::Mat output;
  d->algo->compute( ocv1_gray, ocv2_gray, output );

  return kv::image_container_sptr(
    new kwiver::arrows::ocv::image_container(
      output, kwiver::arrows::ocv::image_container::BGR_COLOR ) );
}

} //end namespace viame
