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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief Implementation for example_detector
 */

#include "example_detector.h"


namespace kwiver {
namespace arrows {
namespace core {

class example_detector::priv
{
public:
  // -- CONSTRUCTORS --
  priv()
  : m_center_x(100.0)
  , m_center_y(100.0)
  , m_height(200.0)
  , m_width(200.0)
  , m_dx(0.0)
  , m_dy(0.0)
  , m_frame_ct(0)
  {}

  ~priv()
  {}

  double m_center_x;
  double m_center_y;
  double m_height;
  double m_width;
  double m_dx;
  double m_dy;
  int m_frame_ct;
}; // end class example_detector::priv


// =============================================================================
example_detector::
example_detector()
        : d( new priv )
{ }


example_detector::
~example_detector()
{ }


// ------------------------------------------------------------------
vital::config_block_sptr
example_detector::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "center_x", d->m_center_x, "Bounding box center x coordinate." );
  config->set_value( "center_y", d->m_center_y, "Bounding box center y coordinate." );
  config->set_value( "height", d->m_height, "Bounding box height." );
  config->set_value( "width", d->m_width, "Bounding box width." );
  config->set_value( "dx", d->m_dx, "Bounding box x translation per frame." );
  config->set_value( "dy", d->m_dy, "Bounding box y translation per frame." );

  return config;
}


// ------------------------------------------------------------------
void
example_detector::
set_configuration(vital::config_block_sptr config_in)
{
  auto config = get_configuration();
  config->merge_config(config_in);

  d->m_center_x     = config->get_value<double>( "center_x" );
  d->m_center_y     = config->get_value<double>( "center_y" );
  d->m_height       = config->get_value<double>( "height" );
  d->m_width        = config->get_value<double>( "width" );
  d->m_dx           = config->get_value<double>( "dx" );
  d->m_dy           = config->get_value<double>( "dy" );
}


// ------------------------------------------------------------------
bool
example_detector::
check_configuration(vital::config_block_sptr config) const
{
  return true;
}


// ------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
example_detector::
detect( vital::image_container_sptr image_data) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set>();

  const double ct = (double)d->m_frame_ct;

  kwiver::vital::bounding_box_d bbox(
          d->m_center_x + ct*d->m_dx - d->m_width/2.0,
          d->m_center_y + ct*d->m_dy - d->m_height/2.0,
          d->m_center_x + ct*d->m_dx + d->m_width/2.0,
          d->m_center_y + ct*d->m_dy + d->m_height/2.0);

  ++d->m_frame_ct;

  auto dot = std::make_shared< kwiver::vital::detected_object_type >();
  dot->set_score( "detection", 1.0 );

  detected_set->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );

  return detected_set;
}

} } } // end namespace
