// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation for hough_circle_detector

#include "hough_circle_detector.h"

#include <viame/algorithm_framework/config/config_difference.h>

#include <viame/opencv_bridge/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

namespace kwiver {

namespace arrows {

namespace ocv {

// ----------------------------------------------------------------------------
bool
hough_circle_detector
::check_configuration( vital::config_block_sptr config_in ) const
{
  vital::config_block_sptr config = this->get_configuration();

  kwiver::vital::config_difference cd( config, config_in );
  return !cd.warn_extra_keys( logger() );
}

// ----------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
hough_circle_detector
::detect( vital::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();

  using namespace kwiver::arrows::ocv;

  cv::Mat src = image_container::vital_to_ocv(
    image_data->get_image(),
    image_container::BGR_COLOR );
  cv::Mat src_gray;

  // Convert it to gray
  cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );

  // Reduce the noise so we avoid false circle detection
  cv::GaussianBlur( src_gray, src_gray, cv::Size( 9, 9 ), 2, 2 );

  std::vector< cv::Vec3f > circles;

  // Apply the Hough Transform to find the circles
  cv::HoughCircles(
    src_gray,                 // i: source image
    circles,                  // o: detected circles
    cv::HOUGH_GRADIENT,       // i: method
    this->get_dp(),           // i: dp
    this->get_min_dist(),     // + src_gray.rows / 8, // i: minDist
    this->get_param1(),       // i: param1 for canny edge detector
    this->get_param2(),       // i: param2 for canny edge detector
    this->get_min_radius(),   // i: min radius
    this->get_max_radius() ); // i: max radius

  LOG_DEBUG( logger(), "Detected " << circles.size() << " objects." );

  // process results
  for( size_t i = 0; i < circles.size(); ++i )
  {
    // Center point [circles[i][0], circles[i][1]]
    // Radius circles[i][2]

    // Bounding box is center +/- radius
    kwiver::vital::bounding_box_d bbox( circles[ i ][ 0 ] - circles[ i ][ 2 ],
      circles[ i ][ 1 ] - circles[ i ][ 2 ],
      circles[ i ][ 0 ] + circles[ i ][ 2 ],
      circles[ i ][ 1 ] + circles[ i ][ 2 ] );

    auto dot = std::make_shared< kwiver::vital::detected_object_type >();
    dot->set_score( "circle", 1.0 );

    detected_set->add(
      std::make_shared< kwiver::vital::detected_object >(
        bbox,
        1.0, dot ) );
  } // end for

  return detected_set;
}

} // namespace ocv

} // namespace arrows

}     // end namespace
