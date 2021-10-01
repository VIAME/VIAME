/*
 * INSERT COPYRIGHT STATEMENT OR DELETE THIS
 */

#include "example_detector.h"

#include <cmath>
#include <iostream>

namespace viame {

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class example_detector::priv
{
public:

  priv() : m_text( "External Plugin C++ Example" ) {}
  ~priv() {}

  std::string m_text;
}; // end class example_detector::priv

// =================================================================================================

example_detector
::example_detector()
  : d( new priv )
{}


example_detector
::  ~example_detector()
{}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
example_detector
::get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value( "text", d->m_text, "Text to display to user." );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
example_detector
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_text = config->get_value< std::string >( "text" );
}


// -------------------------------------------------------------------------------------------------
bool
example_detector
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  if( d->m_text.empty() )
  {
    return false;
  }

  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
example_detector
::detect( kwiver::vital::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();

  std::cout << "Text: " << d->m_text << std::endl;

  return detected_set;
}


} // end namespace
