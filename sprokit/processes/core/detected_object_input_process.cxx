// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for detected_object_set_input process
 */

#include "detected_object_input_process.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/algo/detected_object_set_input.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( file_name, std::string, "", "Name of the detection set file to read." );

create_algorithm_name_config_trait( reader )

//----------------------------------------------------------------
// Private implementation class
class detected_object_input_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;

  algo::detected_object_set_input_sptr m_reader;
}; // end priv class

// ================================================================

detected_object_input_process
::detected_object_input_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new detected_object_input_process::priv )
{
  make_ports();
  make_config();
}

detected_object_input_process
::~detected_object_input_process()
{
}

// ----------------------------------------------------------------
void detected_object_input_process
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

  // Get algo config entries
  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  // validate configuration
  if ( ! algo::detected_object_set_input::check_nested_algo_configuration_using_trait(
         reader, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  algo::detected_object_set_input::set_nested_algo_configuration_using_trait(
    reader,
    algo_config,
    d->m_reader);
  if ( ! d->m_reader )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
             "Unable to create reader." );
  }
}

// ----------------------------------------------------------------
void detected_object_input_process
::_init()
{
  scoped_init_instrumentation();

  d->m_reader->open( d->m_file_name ); // throws
}

// ----------------------------------------------------------------
void detected_object_input_process
::_step()
{
  std::string image_name;
  kwiver::vital::detected_object_set_sptr set;
  bool result(false);
  {
    scoped_step_instrumentation();

    result = d->m_reader->read_set( set, image_name );
  }

  if ( result )
  {
    push_to_port_using_trait( image_file_name, image_name );
    push_to_port_using_trait( detected_object_set, set );
  }
  else
  {
    LOG_DEBUG( logger(), "End of input reached, process terminating" );

    // indicate done
    mark_process_as_complete();
    const sprokit::datum_t dat= sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( detected_object_set, dat );
    push_datum_to_port_using_trait( image_file_name, dat );
  }
}

// ----------------------------------------------------------------
void detected_object_input_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;

  declare_output_port_using_trait( image_file_name, optional );
  declare_output_port_using_trait( detected_object_set, optional );
}

// ----------------------------------------------------------------
void detected_object_input_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( reader );
}

// ================================================================
detected_object_input_process::priv
::priv()
{
}

detected_object_input_process::priv
::~priv()
{
}

} // end namespace
