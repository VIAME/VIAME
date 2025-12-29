// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "burnout_image_enhancer.h"

#include <string>
#include <sstream>
#include <exception>

#include <arrows/vxl/image_container.h>

#include <vital/exceptions.h>

#include <video_transforms/video_enhancement_process.h>

namespace viame {

typedef vidtk::video_enhancement_process< vxl_byte > process_8bit;
typedef vidtk::video_enhancement_process< vxl_uint_16 > process_16bit;

// ----------------------------------------------------------------------------
class burnout_image_enhancer::priv
{
public:
  priv()
    : m_config_file( "burnout_enhancer.conf" )
    , is_16bit_mode( true )
  {}

  ~priv()
  {}

  // Items from the config
  std::string m_config_file;

  // VIDTK config block with parameters
  vidtk::config_block m_vidtk_config;
  bool is_16bit_mode;

  // Pointer to VIDTK process
  std::unique_ptr< vidtk::process > m_process;

  // Configure internal process
  void configure_process( bool for_16bit = false );

  vital::logger_handle_t m_logger;
};

// ----------------------------------------------------------------------------
burnout_image_enhancer
::burnout_image_enhancer()
  : d( new priv() )
{
}

burnout_image_enhancer
::~burnout_image_enhancer()
{
}

// ----------------------------------------------------------------------------

void
burnout_image_enhancer::priv
::configure_process( bool for_16bit )
{
  if( for_16bit == is_16bit_mode )
  {
    return;
  }

  if( for_16bit )
  {
    process_16bit* process = new process_16bit( "filter" );
    m_process.reset( process );
    is_16bit_mode = true;

    if( !process->set_params( m_vidtk_config ) )
    {
      throw std::runtime_error(  "Failed to set pipeline parameters" );
    }

    if( !process->initialize() )
    {
      throw std::runtime_error( "Failed to initialize pipeline" );
    }
  }
  else
  {
    process_8bit* process = new process_8bit( "filter" );
    m_process.reset( process );
    is_16bit_mode = false;

    if( !process->set_params( m_vidtk_config ) )
    {
      throw std::runtime_error(  "Failed to set pipeline parameters" );
    }

    if( !process->initialize() )
    {
      throw std::runtime_error( "Failed to initialize pipeline" );
    }
  }
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
burnout_image_enhancer
::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "config_file", d->m_config_file,  "Name of config file." );

  return config;
}

// ----------------------------------------------------------------------------
void
burnout_image_enhancer
::set_configuration( vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_config_file = config->get_value< std::string >( "config_file" );
  d->m_vidtk_config = process_8bit( "filter" ).params();
  d->m_vidtk_config.parse( d->m_config_file );

  d->configure_process();
}

// ----------------------------------------------------------------------------
bool
burnout_image_enhancer
::check_configuration( vital::config_block_sptr config ) const
{
  std::string config_fn = config->get_value< std::string >( "config_file" );

  if( config_fn.empty() )
  {
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
vital::image_container_sptr
burnout_image_enhancer
::filter( vital::image_container_sptr image_data )
{
  // Configure for correct bit depth
  if( !image_data )
  {
    LOG_WARN( logger(), "Empty image received" );
    return vital::image_container_sptr();
  }

  bool is_16bit = ( image_data->get_image().pixel_traits().num_bytes > 1 );
  d->configure_process( is_16bit );

  // Convert inputs to burnout style inputs, run through process, return output
  kwiver::vital::image_container_sptr output;

  if( is_16bit )
  {
    vil_image_view< vxl_uint_16 > input_image =
      vxl::image_container::vital_to_vxl( image_data->get_image() );

    process_16bit* process = dynamic_cast< process_16bit* >( d->m_process.get() );

    process->set_source_image( input_image );

    if( !process->step() )
    {
      throw std::runtime_error( "Unable to step burnout filter process" );
    }

    output = kwiver::vital::image_container_sptr(
      new arrows::vxl::image_container( process->copied_output_image() ) );
  }
  else
  {
    vil_image_view< vxl_byte > input_image =
      vxl::image_container::vital_to_vxl( image_data->get_image() );

    process_8bit* process = dynamic_cast< process_8bit* >( d->m_process.get() );

    process->set_source_image( input_image );

    if( !process->step() )
    {
      throw std::runtime_error( "Unable to step burnout filter process" );
    }

    output = kwiver::vital::image_container_sptr(
      new arrows::vxl::image_container( process->copied_output_image() ) );
  }

  return output;
}

} // end namespace viame
