/**
 * \file
 * \brief Pass frame in min max index limits
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

//------------------------------------------------------------------------------
// Private implementation class
class filter_frame_index_process::priv
{
public:
  priv();
  ~priv();

  // Configuration settings
  unsigned m_min_frame_count;
  unsigned m_max_frame_count;
};

// =============================================================================
filter_frame_index_process::priv
::priv()
  : m_min_frame_count( 0 )
  , m_max_frame_count( 0 )
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
}


// -----------------------------------------------------------------------------
void
filter_frame_index_process
::_configure()
{
  d->m_min_frame_count = config_value_using_trait( min_frame_count );
  d->m_max_frame_count = config_value_using_trait( max_frame_count );
  
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
  
  if(!d->m_max_frame_count ||
     timestamp.get_frame() >= d->m_min_frame_count && 
     timestamp.get_frame() <= d->m_max_frame_count)
  {
    push_to_port_using_trait( timestamp, timestamp );
    
    image_name = grab_from_port_using_trait( image_file_name );
    push_to_port_using_trait( image_file_name, image_name );
    
    image = grab_from_port_using_trait( image );
    push_to_port_using_trait( image, image );
  }
  
  process::_step();
}


} // end namespace core

} // end namespace viame
