/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "hello_world_detector.h"

#include <cmath>

namespace viame {

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class hello_world_detector::priv
{
public:

  priv() : m_text( "Hello World" ) {}
  ~priv() {}

  std::string m_text;
}; // end class hello_world_detector::priv

// =================================================================================================

hello_world_detector::
hello_world_detector()
  : d( new priv )
{
  attach_logger( "viame.examples.hello_world_detector" );
}


hello_world_detector::
  ~hello_world_detector()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
hello_world_detector::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config =
    kwiver::vital::algo::image_object_detector::get_configuration();

  config->set_value( "text", d->m_text, "Text to display to user." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
hello_world_detector::
set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_text = config->get_value< std::string >( "text" );
}


// -------------------------------------------------------------------------------------------------
bool
hello_world_detector::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  if( d->m_text.empty() )
  {
    LOG_ERROR( logger(), "Text configuration value cannot be empty" );
    return false;
  }

  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
hello_world_detector::
detect( kwiver::vital::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();

  LOG_INFO( logger(), "Text: " << d->m_text );

  return detected_set;
}


} // end namespace
