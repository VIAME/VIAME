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

  std::string algorithm;
  int min_disparity;
  int num_disparities;
  int sad_window_size;
  int block_size;
  int speckle_window_size;
  int speckle_range;

#ifdef VIAME_OPENCV_VER_2
  cv::StereoBM algo;
#else
  cv::Ptr< cv::StereoMatcher > algo;
#endif

  priv()
    : algorithm( "BM" )
    , min_disparity( 0 )
    , num_disparities( 16 )
    , sad_window_size( 21 )
    , block_size( 3 )
    , speckle_window_size( 50 )
    , speckle_range( 5 )
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
::set_configuration( vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->algorithm = config->get_value< std::string >( "algorithm" );
  d->min_disparity = config->get_value< int >( "min_disparity" );
  d->num_disparities = config->get_value< int >( "num_disparities" );
  d->sad_window_size = config->get_value< int >( "sad_window_size" );
  d->block_size = config->get_value< int >( "block_size" );
  d->speckle_window_size = config->get_value< int >( "speckle_window_size" );
  d->speckle_range = config->get_value< int >( "speckle_range" );

#ifdef VIAME_OPENCV_VER_2
  if( d->algorithm == "BM" )
  {
    d->algo.init( 0, d->num_disparities, d->sad_window_size );
  }
  else if( d->algorithm == "SGBM" )
  {
    throw std::runtime_error( "Unable to use type SGBM with OpenCV 2" );
  }
  else
  {
    throw std::runtime_error( "Invalid algorithm type " + d->algorithm );
  }
#else
  if( d->algorithm == "BM" )
  {
    d->algo = cv::StereoBM::create( d->num_disparities, d->sad_window_size );
    d->algo->setSpeckleWindowSize(d->speckle_window_size);
    d->algo->setSpeckleRange (d->speckle_range);
  }
  else if( d->algorithm == "SGBM" )
  {
    d->algo = cv::StereoSGBM::create( d->min_disparity, d->num_disparities, d->block_size );
    d->algo->setSpeckleWindowSize(d->speckle_window_size);
    d->algo->setSpeckleRange (d->speckle_range);
  }
  else
  {
    throw std::runtime_error( "Invalid algorithm type " + d->algorithm );
  }
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
  cv::Mat ocv1 = arrows::ocv::image_container::vital_to_ocv( left_image->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat ocv2 = arrows::ocv::image_container::vital_to_ocv( right_image->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR  );
  
  cv::Mat ocv1_gray, ocv2_gray;

  // Convert each image to grayscale independently (they may have different channel counts)
  if( ocv1.channels() == 3 )
  {
#if CV_MAJOR_VERSION < 4
    cvtColor( ocv1, ocv1_gray, CV_BGR2GRAY );
#else
    cv::cvtColor( ocv1, ocv1_gray, cv::COLOR_BGR2GRAY );
#endif
  }
  else if( ocv1.channels() == 4 )
  {
#if CV_MAJOR_VERSION < 4
    cvtColor( ocv1, ocv1_gray, CV_BGRA2GRAY );
#else
    cv::cvtColor( ocv1, ocv1_gray, cv::COLOR_BGRA2GRAY );
#endif
  }
  else
  {
    ocv1_gray = ocv1;
  }

  if( ocv2.channels() == 3 )
  {
#if CV_MAJOR_VERSION < 4
    cvtColor( ocv2, ocv2_gray, CV_BGR2GRAY );
#else
    cv::cvtColor( ocv2, ocv2_gray, cv::COLOR_BGR2GRAY );
#endif
  }
  else if( ocv2.channels() == 4 )
  {
#if CV_MAJOR_VERSION < 4
    cvtColor( ocv2, ocv2_gray, CV_BGRA2GRAY );
#else
    cv::cvtColor( ocv2, ocv2_gray, cv::COLOR_BGRA2GRAY );
#endif
  }
  else
  {
    ocv2_gray = ocv2;
  }

  cv::Mat output;

#if CV_MAJOR_VERSION == 2
  d->algo( ocv1_gray, ocv2_gray, output );
#else
  d->algo->compute( ocv1_gray, ocv2_gray, output );
#endif

  return kwiver::vital::image_container_sptr( new arrows::ocv::image_container( output,
    kwiver::arrows::ocv::image_container::BGR_COLOR ) );
}

} //end namespace viame
