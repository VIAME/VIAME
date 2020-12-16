// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for hough_circle_detector
 */

#include "hough_circle_detector.h"

#include <vital/config/config_difference.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

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
    : m_dp(1)
    , m_min_dist(100)
    , m_param1(200)
    , m_param2(100)
    , m_min_radius(0)
    , m_max_radius(0)
  {}

  ~priv()
  {}

  // Hough algorithm parameters
  double m_dp;
  double m_min_dist;
  double m_param1;
  double m_param2;
  int m_min_radius;
  int m_max_radius;

}; // end class hough_circle_detector::priv

  // ==================================================================
hough_circle_detector::
hough_circle_detector()
  : d( new priv )
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

  config->set_value( "dp", d->m_dp,
                     "Inverse ratio of the accumulator resolution to the image resolution. "
                     "For example, if dp=1 , the accumulator has the same resolution as the input image. "
                     "If dp=2 , the accumulator has half as big width and height." );

  config->set_value( "min_dist", d->m_min_dist,
                     "Minimum distance between the centers of the detected circles. "
                     "If the parameter is too small, multiple neighbor circles may be falsely "
                     "detected in addition to a true one. If it is too large, some circles may be missed." );

  config->set_value( "param1", d->m_param1,
                     "First method-specific parameter. In case of CV_HOUGH_GRADIENT , "
                     "it is the higher threshold of the two passed to the Canny() edge detector "
                     "(the lower one is twice smaller)." );

  config->set_value( "param2", d->m_param2,
                     "Second method-specific parameter. In case of CV_HOUGH_GRADIENT , "
                     "it is the accumulator threshold for the circle centers at the detection stage. "
                     "The smaller it is, the more false circles may be detected. Circles, "
                     "corresponding to the larger accumulator values, will be returned first." );

  config->set_value( "min_radius", d->m_min_radius, "Minimum circle radius." );

  config->set_value( "max_radius", d->m_max_radius, "Maximum circle radius." );

  return config;
}

// ------------------------------------------------------------------
void
hough_circle_detector::
set_configuration(vital::config_block_sptr config_in)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  kwiver::vital::config_difference cd( config, config_in );
  cd.warn_extra_keys( logger() );

  config->merge_config( config_in );

  d->m_dp         = config->get_value<double>( "dp" );
  d->m_min_dist   = config->get_value<double>( "min_dist" );
  d->m_param1     = config->get_value<double>( "param1" );
  d->m_param2     = config->get_value<double>( "param2" );
  d->m_min_radius = config->get_value<int>( "min_radius" );
  d->m_max_radius = config->get_value<int>( "max_radius" );
}

// ------------------------------------------------------------------
bool
hough_circle_detector::
check_configuration(vital::config_block_sptr config_in) const
{
  vital::config_block_sptr config = this->get_configuration();

  kwiver::vital::config_difference cd( config, config_in );
  return ! cd.warn_extra_keys( logger() );
}

// ------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
hough_circle_detector::
detect( vital::image_container_sptr image_data) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set>();

  using namespace kwiver::arrows::ocv;
  cv::Mat src = image_container::vital_to_ocv( image_data->get_image(),
                                               image_container::BGR_COLOR );
  cv::Mat src_gray;

  // Convert it to gray
  cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );

  // Reduce the noise so we avoid false circle detection
  cv::GaussianBlur( src_gray, src_gray, cv::Size( 9, 9 ), 2, 2 );

  std::vector< cv::Vec3f > circles;

  // Apply the Hough Transform to find the circles
  cv::HoughCircles( src_gray, // i: source image
                    circles, // o: detected circles
                    cv::HOUGH_GRADIENT, // i: method
                    d->m_dp, // i: dp
                    d->m_min_dist, //+ src_gray.rows / 8, // i: minDist
                    d->m_param1, // i: param1 for canny edge detector
                    d->m_param2, // i: param2 for canny edge detector
                    d->m_min_radius, // i: min radius
                    d->m_max_radius ); // i: max radius

  LOG_DEBUG( logger(), "Detected " << circles.size() << " objects." );

  // process results
  for ( size_t i = 0; i < circles.size(); ++i )
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
