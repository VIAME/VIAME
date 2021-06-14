// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for read_object_track_set process
 */

#include "read_object_track_process.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/algo/read_object_track_set.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

create_config_trait( file_name, std::string, "",
  "Name of the track descriptor set file to read." );

create_algorithm_name_config_trait( reader );

//--------------------------------------------------------------------------------
// Private implementation class
class read_object_track_process::priv
{
public:
  priv() : m_reader_finished( false ) {}
  ~priv() {}

  // Configuration values
  std::string m_file_name;
  bool m_reader_finished;

  algo::read_object_track_set_sptr m_reader;
}; // end priv class

// ===============================================================================

read_object_track_process
::read_object_track_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new read_object_track_process::priv )
{
  set_data_checking_level( check_sync );

  make_ports();
  make_config();
}

read_object_track_process
::~read_object_track_process()
{
}

// -------------------------------------------------------------------------------
void read_object_track_process
::_configure()
{
  // Get process config entries
  d->m_file_name = config_value_using_trait( file_name );

  if( d->m_file_name.empty() )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
      "Required file name not specified." );
  }

  // Get algo config entries
  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  // validate configuration
  if( ! algo::read_object_track_set::check_nested_algo_configuration_using_trait(
          reader, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  algo::read_object_track_set::set_nested_algo_configuration_using_trait(
    reader,
    algo_config,
    d->m_reader );

  if( !d->m_reader )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Unable to create reader." );
  }
}

// -------------------------------------------------------------------------------
void read_object_track_process
::_init()
{
  d->m_reader->open( d->m_file_name ); // throws
}

// -------------------------------------------------------------------------------
void read_object_track_process
::_step()
{
  bool end_process = false;

  std::string file_name;
  kwiver::vital::object_track_set_sptr set;

  if( has_input_port_edge_using_trait( image_file_name ) )
  {
    auto port_info = peek_at_port_using_trait( image_file_name );

    if( port_info.datum->type() == sprokit::datum::complete )
    {
      end_process = true;
    }
    else
    {
      file_name = grab_from_port_using_trait( image_file_name );
    }
  }

  if( !end_process && !d->m_reader_finished && !d->m_reader->read_set( set ) )
  {
    // Indicates the reader is done producing and tracks and won't produce more.
    d->m_reader_finished = true;

    // If false, we are driven by an external frame source which might continue
    // to send frames after ones which don't contain tracks so we don't want to
    // send a complete message yet and instead rely on that source telling us
    // when we're done.
    if( file_name.empty() )
    {
      end_process = true;
    }
  }

  if( end_process )
  {
    LOG_DEBUG( logger(), "End of input reached, process terminating" );
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();
    push_datum_to_port_using_trait( object_track_set, dat );
  }
  else
  {
    if( !set )
    {
      set = std::make_shared< kwiver::vital::object_track_set >();
    }
    push_to_port_using_trait( object_track_set, set );
  }
}

// -------------------------------------------------------------------------------
void read_object_track_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;

  declare_input_port_using_trait( image_file_name, optional );
  declare_output_port_using_trait( object_track_set, optional );
}

// -------------------------------------------------------------------------------
void read_object_track_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( reader );
}


} // end namespace
