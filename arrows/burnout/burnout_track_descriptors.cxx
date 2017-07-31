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

#include "burnout_track_descriptors.h"

#include <string>
#include <sstream>
#include <exception>


namespace kwiver {
namespace arrows {
namespace burnout {

// ==================================================================
class burnout_track_descriptors::priv
{
public:
  priv()
    : m_config( "" )
  {}

  ~priv()
  {
  }

  // Items from the config
  std::string m_config;

  kwiver::vital::logger_handle_t m_logger;
};


// ==================================================================
burnout_track_descriptors::
burnout_track_descriptors()
  : d( new priv() )
{

}


burnout_track_descriptors::
~burnout_track_descriptors()
{}


// --------------------------------------------------------------------
vital::config_block_sptr
burnout_track_descriptors::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "config", d->m_config,
    "Name of config file." );

  return config;
}


// --------------------------------------------------------------------
void
burnout_track_descriptors::
set_configuration( vital::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing
  // a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  this->d->m_config = config->get_value< std::string >( "config" );
}


// --------------------------------------------------------------------
bool
burnout_track_descriptors::
check_configuration( vital::config_block_sptr config ) const
{
  std::string config_fn = config->get_value< std::string >( "config" );

  if( config_fn.empty() )
  {
    return false;
  }

  return true;
}


// --------------------------------------------------------------------
kwiver::vital::track_descriptor_set_sptr
burnout_track_descriptors::
compute( kwiver::vital::image_container_sptr image_data,
         kwiver::vital::object_track_set_sptr tracks )
{

  return kwiver::vital::track_descriptor_set_sptr();
}


} } } // end namespace
