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

#include "uw_predictor_classifier.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>

#include <cmath>

namespace viame {

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class uw_predictor_classifier::priv
{
public:

  priv() {}
  ~priv() {}

  std::string m_config_file;
};

// =================================================================================================

uw_predictor_classifier::
uw_predictor_classifier()
  : d( new priv )
{}


uw_predictor_classifier::
  uw_predictor_classifier( const uw_predictor_classifier& other )
  : d( new priv( *other.d ) )
{}


uw_predictor_classifier::
  ~uw_predictor_classifier()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
uw_predictor_classifier::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value( "config_file", d->m_config_file,
                     "Name of uw_predictor configuration file." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
uw_predictor_classifier::
set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_config_file = config->get_value< std::string >( "config_file" );
}


// -------------------------------------------------------------------------------------------------
bool
uw_predictor_classifier::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
uw_predictor_classifier::
detect( kwiver::vital::image_container_sptr image_data ) const
{
  auto input_detections = std::make_shared< kwiver::vital::detected_object_set >(); // TODO become input detections
  auto output_detections = std::make_shared< kwiver::vital::detected_object_set >();

  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );

  // process results
  VITAL_FOREACH( auto det, input_detections->select() )
  {
    // Crop out chip
    // []

    // Run UW predictor code on matlab chip
    // []

    // Convert UW detections to KWIVER format
    //auto dot = std::make_shared< kwiver::vital::detected_object_type >( det.classIDs, det.classProbabilities );

    // Create detection
    //detected_set->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
  }

  return output_detections;
}


} // end namespace
