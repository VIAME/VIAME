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

#include "camtrawl_detect.h"

#include <arrows/ocv/image_container.h>
//#include <opencv2/core/core.hpp>
#include <vital/vital_foreach.h>

#include <cmath>
#include <iostream>

namespace viame {

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class camtrawl_detect::priv
{
public:

  priv() {}
  ~priv() {}

}; // end class camtrawl_detect::priv

// =================================================================================================

camtrawl_detect::
camtrawl_detect()
  : d( new priv )
{}


camtrawl_detect::
  ~camtrawl_detect()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
camtrawl_detect::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  return config;
}


// -------------------------------------------------------------------------------------------------
void
camtrawl_detect::
set_configuration( kwiver::vital::config_block_sptr config )
{
}


// -------------------------------------------------------------------------------------------------
bool
camtrawl_detect::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
camtrawl_detect::
detect( kwiver::vital::image_container_sptr image_data ) const
{
  //// TODO: IMPLEMENT OR CALL DETECTOR ALGORITHM
  auto vital_img = image_data->get_image();
  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv(vital_img);

  ////kwiver::arrows::ocv::image_container::vital_to_ocv(  );
  //// TODO: do image processing

  //// pack results into expected type
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();
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
    detected_set->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, obj_type ) );
  } // end for


  return detected_set;
}


} // end namespace

