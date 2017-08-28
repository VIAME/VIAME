/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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

#include "camtrawl_biomarkers.h"

#include <cmath>
#include <iostream>

namespace viame {

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class camtrawl_biomarkers::priv
{
public:

  priv() {}
  ~priv() {}

}; // end class camtrawl_biomarkers::priv

// =================================================================================================

camtrawl_biomarkers::
camtrawl_biomarkers()
  : d( new priv )
{}


camtrawl_biomarkers::
  ~camtrawl_biomarkers()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
camtrawl_biomarkers::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();
  //config->set_value( "text", d->m_text, "Text to display to user." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
camtrawl_biomarkers::
set_configuration( kwiver::vital::config_block_sptr config )
{
  //d->m_text = config->get_value< std::string >( "text" );
}


// -------------------------------------------------------------------------------------------------
bool
camtrawl_biomarkers::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  //if( d->m_text.empty() )
  //{
  //  return false;
  //}

  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::feature_set_sptr
camtrawl_biomarkers::
detect( kwiver::vital::detected_object_set_sptr image_data ) const
{
  // TODO: IMPLEMENT OR CALL DETECTOR ALGORITHM

  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );
  // TODO: do image processing

  // pack results into expected type
  auto feature_points = std::make_shared< kwiver::vital::feature_set >();
  for (int i = 0; i < 3; i++)
  {
    double xmin = 0 + i;
    double ymin = 0 + i;
    double xmax = 1 + i;
    double ymax = 1 + i;

    // Create kwiver style bounding box
    kwiver::vital::bounding_box_d bbox(xmin, ymin, xmax, ymax);

    std::vector< std::string > class_names = {"redfish", "bluefish"};
    std::vector< double > class_probs = {.9, .1};

    // Create possible object types.
    auto obj_type = std::make_shared< kwiver::vital::detected_object_type >( class_names, class_probs );

    // Create detection
    feature_points->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
  } // end for


  return feature_points;
}


} // end namespace


