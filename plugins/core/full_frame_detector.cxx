/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

/**
 * \file
 * \brief Implementation for full_frame_detector
 */

#include "full_frame_detector.h"

#include <vector>


namespace viame {

namespace kv = kwiver::vital;

/// Private implementation class
class full_frame_detector::priv
{
public:

  /// Constructor
  priv()
  : detection_type( "generic_object_proposal" )
  {
  }

  /// Destructor
  ~priv()
  {
  }

  /// Parameters
  std::string detection_type;
};


/// Constructor
full_frame_detector
::full_frame_detector()
  : d( new priv )
{
}


full_frame_detector
::~full_frame_detector()
{
}


/// Settings
kv::config_block_sptr
full_frame_detector
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::image_object_detector::get_configuration();

  config->set_value( "detection_type", d->detection_type,
                     "Object type to add to newly created detected objects" );

  return config;
}


void
full_frame_detector
::set_configuration(kv::config_block_sptr config)
{
  d->detection_type = config->get_value<std::string>( "detection_type" );
}


bool
full_frame_detector
::check_configuration(kv::config_block_sptr config) const
{
  return true;
}


/// Run full frame descriptor
kv::detected_object_set_sptr
full_frame_detector
::detect( kv::image_container_sptr image_data ) const
{
  auto output = std::make_shared< kv::detected_object_set >();

  if( image_data->height() > 0 && image_data->width() > 0 )
  {
    kv::bounding_box_d det_box( 0, 0,
                                image_data->width(),
                                image_data->height() );

    if( !d->detection_type.empty() )
    {
      auto dot = std::make_shared< kv::detected_object_type >();
      dot->set_score( d->detection_type, 1.0 );

      output->add(
        std::make_shared< kv::detected_object >(
          det_box, 1.0, dot ) );
    }
    else
    {
      output->add(
        std::make_shared< kv::detected_object >(
          det_box, 1.0 ) );
    }
  }

  return output;
}

} // end namespace viame
