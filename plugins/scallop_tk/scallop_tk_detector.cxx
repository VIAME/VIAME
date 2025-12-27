/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "scallop_tk_detector.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>

#include <cmath>

#include <ScallopTK/Pipelines/CoreDetector.h>

namespace viame {

using namespace ScallopTK;

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class scallop_tk_detector::priv
{
public:

  priv() {}
  ~priv() {}

  std::string m_config_file;
  std::shared_ptr< CoreDetector > m_detector;
}; // end class scallop_tk_detector::priv

// =================================================================================================

scallop_tk_detector::
scallop_tk_detector()
  : d( new priv )
{}


scallop_tk_detector::
  ~scallop_tk_detector()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
scallop_tk_detector::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value( "config_file", d->m_config_file,
                     "Name of ScallopTK configuration file." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
scallop_tk_detector::
set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_config_file = config->get_value< std::string >( "config_file" );

  // Create new detector.
  d->m_detector = std::make_shared< CoreDetector >( d->m_config_file );
}


// -------------------------------------------------------------------------------------------------
bool
scallop_tk_detector::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
scallop_tk_detector::
detect( kwiver::vital::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();

  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image(),
    kwiver::arrows::ocv::image_container::RGB_COLOR );

  auto det_list = d->m_detector->processFrame( src );

  LOG_DEBUG( logger(), "Detected " << det_list.size() << " objects." );

  // process results
  for( auto det : det_list )
  {
    // Get parameters from ellipse output
    double angle = det.angle * PI / 180;
    double a = det.major * cos( angle );
    double b = det.minor * sin( angle );
    double c = det.major * sin( angle );
    double d = det.minor * cos( angle );
    double width = sqrt( ( a * a ) + ( b * b ) ) * 2;
    double height = sqrt( ( c * c ) + ( d * d ) ) * 2;
    double x = det.c - width * 0.5;
    double y = det.r - height * 0.5;

    // Create kwiver style bounding box
    kwiver::vital::bounding_box_d bbox( kwiver::vital::bounding_box_d::vector_type( x, y ), width, height );

    // Create possible object types.
    auto dot = std::make_shared< kwiver::vital::detected_object_type >( det.classIDs, det.classProbabilities );

    // Create detection
    detected_set->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
  } // end for

  return detected_set;
}


} // end namespace
