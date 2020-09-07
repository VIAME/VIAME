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

#include "uw_predictor_classifier.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>

#include "util.h"
#include "classHierarchy.h"
#include "SpeciesIDLib.h"

#include <cmath>
#include <string>

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

  std::string m_model_file;
  FishSpeciesID m_fish_model;
};

// =================================================================================================

uw_predictor_classifier::
uw_predictor_classifier()
  : d( new priv )
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

  config->set_value( "model_file", d->m_model_file,
                     "Name of uw_predictor model file." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
uw_predictor_classifier::
set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_model_file = config->get_value< std::string >( "model_file" );

	d->m_fish_model.loadModel( d->m_model_file.c_str() );
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
refine( kwiver::vital::image_container_sptr image_data,
  kwiver::vital::detected_object_set_sptr input_dets ) const
{
  auto output_detections = std::make_shared< kwiver::vital::detected_object_set >();

  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );

  if( src.channels() == 3 )
  {
    cv::cvtColor( src, src, CV_RGB2GRAY );
  }

  // process results
  for( auto det : *input_dets )
  {
    // Crop out chip
    auto bbox = det->bounding_box();

    cv::Rect roi( bbox.min_x(), bbox.min_y(), bbox.width(), bbox.height() );
    cv::Mat roi_crop = src( roi );

    // Run UW predictor code on each chip
    vector< int > predictions;
    vector< double > probabilities;

    cv::Mat segment_chip = kwiver::arrows::ocv::image_container::vital_to_ocv( det->mask()->get_image() );

    if( segment_chip.channels() == 3 )
    {
      cv::cvtColor( segment_chip, segment_chip, CV_RGB2GRAY );
    }

    cv::Mat fg_rect;
    bool is_partial = d->m_fish_model.predict( roi_crop, segment_chip, predictions, probabilities, fg_rect );

    // Convert UW detections to KWIVER format
    vector< string > names;

    for( int i : predictions )
    {
      names.push_back( std::to_string( i ) );
    }

    auto dot = std::make_shared< kwiver::vital::class_map >( names, probabilities );

    // Create detection
    output_detections->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
  }

  return output_detections;
}


} // end namespace
