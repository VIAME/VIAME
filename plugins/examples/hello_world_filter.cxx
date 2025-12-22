/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "hello_world_filter.h"

#include <cmath>

namespace viame {

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class hello_world_filter::priv
{
public:

  priv() : m_text( "Hello World" ) {}
  ~priv() {}

  std::string m_text;
}; // end class hello_world_filter::priv

// =================================================================================================

hello_world_filter::
hello_world_filter()
  : d( new priv )
{
  attach_logger( "viame.examples.hello_world_filter" );
}


hello_world_filter::
  ~hello_world_filter()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
hello_world_filter::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config =
    kwiver::vital::algo::image_filter::get_configuration();

  config->set_value( "text", d->m_text, "Text to display to user." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
hello_world_filter::
set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_text = config->get_value< std::string >( "text" );
}


// -------------------------------------------------------------------------------------------------
bool
hello_world_filter::
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
kwiver::vital::image_container_sptr
hello_world_filter::
filter( kwiver::vital::image_container_sptr image_data )
{
  LOG_INFO( logger(), "Text: " << d->m_text );

  return image_data;
}


} // end namespace
