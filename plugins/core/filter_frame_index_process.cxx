/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Pass frame with step index and in min max limits
 */

#include "filter_frame_index_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( min_frame_count, unsigned, "0",
  "If set, Require frame index higher than to pass frame" );
create_config_trait( max_frame_count, unsigned, "0",
  "If set, Require frame index lower than to pass frame" );
create_config_trait( frame_step, unsigned, "0",
  "If set, Pass frame at each frame step" );

//------------------------------------------------------------------------------
// Private implementation class
class filter_frame_index_process::priv
{
public:
  priv();
  ~priv();

  // Internal variable
  kv::frame_id_t m_last_frame_id;
  bool m_first_frame;
  
  // Configuration settings
  kv::frame_id_t m_min_frame_count;
  kv::frame_id_t m_max_frame_count;
  kv::frame_id_t m_frame_step;
};

// =============================================================================
filter_frame_index_process::priv
::priv()
  : m_last_frame_id(0)
  , m_first_frame(true)
  , m_min_frame_count( 0 )
  , m_max_frame_count( 0 )
  , m_frame_step( 0 )
{
}


filter_frame_index_process::priv
::~priv()
{
}

// =============================================================================

filter_frame_index_process
::filter_frame_index_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new filter_frame_index_process::priv() )
{
  make_ports();
  make_config();
}


filter_frame_index_process
::~filter_frame_index_process()
{
}


// -----------------------------------------------------------------------------
void
filter_frame_index_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );  
  declare_input_port_using_trait( image_file_name, required );
  declare_input_port_using_trait( image, required );

  // -- output --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( image_file_name, optional );
  declare_output_port_using_trait( image, optional );
}


// -----------------------------------------------------------------------------
void
filter_frame_index_process
::make_config()
{
  declare_config_using_trait( min_frame_count );
  declare_config_using_trait( max_frame_count );
  declare_config_using_trait( frame_step );
}


// -----------------------------------------------------------------------------
void
filter_frame_index_process
::_configure()
{
  d->m_last_frame_id = 0;
  d->m_first_frame = true;
  d->m_min_frame_count = config_value_using_trait( min_frame_count );
  d->m_max_frame_count = config_value_using_trait( max_frame_count );
  d->m_frame_step = config_value_using_trait( frame_step );
  
  if ( d->m_min_frame_count > d->m_max_frame_count )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Invalid min/max frame index limits" );
  }
}


// -----------------------------------------------------------------------------
void
filter_frame_index_process
::_step()
{
  kv::timestamp timestamp;
  std::string image_name;
  kv::image_container_sptr image;

  timestamp = grab_from_port_using_trait( timestamp );
  
  if( ( !d->m_max_frame_count && !d->m_frame_step ) ||
      ( timestamp.get_frame() >= d->m_min_frame_count &&
        timestamp.get_frame() <= d->m_max_frame_count ) )
  {
    if( d->m_first_frame ||
        ( timestamp.get_frame() - d->m_last_frame_id ) >= d->m_frame_step )
    {
      push_to_port_using_trait( timestamp, timestamp );
      image_name = grab_from_port_using_trait( image_file_name );
      push_to_port_using_trait( image_file_name, image_name );
      image = grab_from_port_using_trait( image );
      push_to_port_using_trait( image, image );
      
      d->m_first_frame = false;
      d->m_last_frame_id = timestamp.get_frame();
    }
    else
    {
      push_to_port_using_trait( image, kwiver::vital::image_container_sptr() );
    }
    
  }
  
//  process::_step();
}


} // end namespace core

} // end namespace viame
