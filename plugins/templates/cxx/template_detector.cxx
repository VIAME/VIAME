/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "@template@_detector.h"

#include <cmath>

namespace viame {

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class @template@_detector::priv
{
public:

  priv() : m_text( "Hello World" ) {}
  ~priv() {}

  std::string m_text;
}; // end class @template@_detector::priv

// =================================================================================================

@template@_detector
::@template@_detector()
  : d( new priv )
{}


@template@_detector
::  ~@template@_detector()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
@template@_detector
::get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  //++ Add configuration items to the config block that are needed for this algorithm
  config->set_value( "text", d->m_text, "Text to display to user." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
@template@_detector
::set_configuration( kwiver::vital::config_block_sptr config )
{
  //++ Get configuration items from 'config' block
  d->m_text = config->get_value< std::string >( "text" );
}


// -------------------------------------------------------------------------------------------------
bool
@template@_detector
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  //++ check for condition that would prevent the detector from running correctly.
  //++ Not necessarily limited to config related problems.
  if( d->m_text.empty() )
  {
    return false;
  }

  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
@template@_detector
::detect( kwiver::vital::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();

  //++ insert detector code here
  LOG_INFO( logger(), "Text: " << d->m_text );

  return detected_set;
}

} // end namespace
