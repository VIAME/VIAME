/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

#include "hough_circle_detector.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class hough_circle_detector::priv
{
public:
  // -- CONSTRUCTORS --
  priv()
  {}

  ~priv()
  {}

}; // end class hough_circle_detector::priv


hough_circle_detector::
hough_circle_detector()
  : d( new priv )
{
}


hough_circle_detector::
hough_circle_detector( const hough_circle_detector& other)
  : d( new priv( *other.d ) )
{ }


 hough_circle_detector::
~hough_circle_detector()
{ }


// ------------------------------------------------------------------
vital::config_block_sptr
hough_circle_detector::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  return config;
}


// ------------------------------------------------------------------
void
hough_circle_detector::
set_configuration(vital::config_block_sptr config)
{ }


// ------------------------------------------------------------------
bool
hough_circle_detector::
check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// ------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
hough_circle_detector::
detect( vital::image_container_sptr image_data) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set>();

  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );
  cv::Mat src_gray;

  // Convert it to gray
  cvtColor( src, src_gray, CV_BGR2GRAY );

  // Reduce the noise so we avoid false circle detection
  cv::GaussianBlur( src_gray, src_gray, cv::Size( 9, 9 ), 2, 2 );

  cv::vector< cv::Vec3f > circles;

  // Apply the Hough Transform to find the circles
  cv::HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0 );

  // process results
  for ( size_t i = 0; i < circles.size(); i++ )
  {
    // Center point [circles[i][0], circles[i][1]]
    // Radius circles[i][2]

    // Bounding box is center +/- radius
    kwiver::vital::bounding_box_d bbox( circles[i][0] - circles[i][2], circles[i][1] - circles[i][2],
                                        circles[i][0] + circles[i][2], circles[i][1] + circles[i][2] );

    auto dot = std::make_shared< kwiver::vital::detected_object_type >();
    dot->set_score( "circle", 1.0 );

    detected_set->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
  } // end for

  return detected_set;
}

} } } // end namespace
