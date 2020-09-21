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

#include "burnout_pixel_classification.h"

#include <string>
#include <sstream>
#include <exception>

#include <arrows/vxl/image_container.h>

#include <vital/exceptions.h>

#include <pipelines/remove_burnin_pipeline.h>


namespace kwiver {
namespace arrows {
namespace burnout {


// ==================================================================================
class burnout_pixel_classification::priv
{
public:
  priv()
    : m_config_file( "burnout_classification.conf" )
    , m_output_type( "mask" )
    , m_process( "classifier" )
  {}

  ~priv()
  {}

  // Items from the config
  std::string m_config_file;
  std::string m_output_type;

  vidtk::remove_burnin_pipeline< vxl_byte > m_process;
  vital::logger_handle_t m_logger;
};


// ==================================================================================
burnout_pixel_classification
::burnout_pixel_classification()
  : d( new priv() )
{
}


burnout_pixel_classification
::~burnout_pixel_classification()
{}


// ----------------------------------------------------------------------------------
vital::config_block_sptr
burnout_pixel_classification
::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "config_file", d->m_config_file,  "Name of config file." );

  config->set_value( "output_type", d->m_output_type,  "Type of output."
    "Can be set to either \"inpainted_image\" or \"mask\" depending on if you want "
    "a mask showing pixel level detections, or an inpainted output image with "
    "all masked pixels filled in with other values." );

  return config;
}


// ----------------------------------------------------------------------------------
void
burnout_pixel_classification
::set_configuration( vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_config_file = config->get_value< std::string >( "config_file" );
  d->m_output_type = config->get_value< std::string >( "output_type" );

  vidtk::config_block vidtk_config = d->m_process.params();
  vidtk_config.parse( d->m_config_file );

  if( !d->m_process.set_params( vidtk_config ) )
  {
    std::string reason = "Failed to set pipeline parameters";
    VITAL_THROW( vital::algorithm_configuration_exception, type_name(), impl_name(), reason );
  }

  if( !d->m_process.initialize() )
  {
    std::string reason = "Failed to initialize pipeline";
    VITAL_THROW( vital::algorithm_configuration_exception, type_name(), impl_name(), reason );
  }
}


// ----------------------------------------------------------------------------------
bool
burnout_pixel_classification
::check_configuration( vital::config_block_sptr config ) const
{
  std::string config_fn = config->get_value< std::string >( "config_file" );

  if( config_fn.empty() )
  {
    return false;
  }

  std::string output_type = config->get_value< std::string >( "output_type" );

  if( output_type != "mask" && output_type != "inpainted_image" )
  {
    return false;
  }

  return true;
}


// ----------------------------------------------------------------------------------
vital::image_container_sptr
burnout_pixel_classification
::filter( vital::image_container_sptr image_data )
{
  // Convert inputs to burnout style inputs
  vil_image_view< vxl_byte > input_image;

  if( image_data )
  {
    input_image = vxl::image_container::vital_to_vxl( image_data->get_image() );
  }
  else
  {
    return vital::image_container_sptr();
  }

  // Process imagery
  d->m_process.set_image( input_image );

  if( !d->m_process.step() )
  {
    throw std::runtime_error( "Unable to step burnout filter process" );
  }

  // Return output in KWIVER wrapper
  if( d->m_output_type == "mask" )
  {
    return kwiver::vital::image_container_sptr(
      new arrows::vxl::image_container( d->m_process.detected_mask() ) );
  }

  return kwiver::vital::image_container_sptr(
    new arrows::vxl::image_container( d->m_process.inpainted_image() ) );
}


} } } // end namespace
