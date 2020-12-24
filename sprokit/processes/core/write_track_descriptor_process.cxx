// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for write_track_descriptor_set process
 */

#include "write_track_descriptor_process.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/algo/write_track_descriptor_set.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( file_name, std::string, "",
  "Name of the track descriptor set file to write." );

create_algorithm_name_config_trait( writer );

//--------------------------------------------------------------------------------
// Private implementation class
class write_track_descriptor_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;

  algo::write_track_descriptor_set_sptr m_writer;
}; // end priv class

// ===============================================================================

write_track_descriptor_process
::write_track_descriptor_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new write_track_descriptor_process::priv )
{
  make_ports();
  make_config();
}

write_track_descriptor_process
::~write_track_descriptor_process()
{
}

// -------------------------------------------------------------------------------
void write_track_descriptor_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get process config entries
  d->m_file_name = config_value_using_trait( file_name );
  if ( d->m_file_name.empty() )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
             "Required file name not specified." );
  }

  // Get algo conrig entries
  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  // validate configuration
  if ( ! algo::write_track_descriptor_set::check_nested_algo_configuration_using_trait(
         writer, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  algo::write_track_descriptor_set::set_nested_algo_configuration_using_trait(
    writer,
    algo_config,
    d->m_writer);
  if ( ! d->m_writer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
             "Unable to create writer." );
  }
}

// -------------------------------------------------------------------------------
void write_track_descriptor_process
::_init()
{
  scoped_init_instrumentation();

  d->m_writer->open( d->m_file_name ); // throws
}

// -------------------------------------------------------------------------------
void write_track_descriptor_process
::_step()
{
  std::string file_name;

  // image name is optional
  if ( has_input_port_edge_using_trait( image_file_name ) )
  {
    file_name = grab_from_port_using_trait( image_file_name );
  }

  kwiver::vital::track_descriptor_set_sptr input
    = grab_from_port_using_trait( track_descriptor_set );

  {
    scoped_step_instrumentation();
    d->m_writer->write_set( input );
  }
}

// -------------------------------------------------------------------------------
void write_track_descriptor_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( image_file_name, optional );
  declare_input_port_using_trait( track_descriptor_set, required );
}

// -------------------------------------------------------------------------------
void write_track_descriptor_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( writer );
}

// ===============================================================================
write_track_descriptor_process::priv
::priv()
{
}

write_track_descriptor_process::priv
::~priv()
{
}

} // end namespace
