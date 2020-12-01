// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for example_detector
 */

#include "example_detector.h"

#include <vital/vital_config.h>

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
check_configuration( VITAL_UNUSED vital::config_block_sptr config) const
{
  return true;
}

// ------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
example_detector::
detect( VITAL_UNUSED vital::image_container_sptr image_data) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set>();
  const double ct = static_cast<double>(d->m_frame_ct);

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
