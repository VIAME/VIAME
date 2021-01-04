// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for detected_object_set_output process
 */

#include "detected_object_output_process.h"

#include <vital/vital_types.h>
#include <vital/exceptions.h>
#include <vital/algo/detected_object_set_output.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( file_name, std::string, "", "Name of the detection set file to write." );
create_algorithm_name_config_trait( writer );

/**
 * \class detected_object_output_process
 *
 * \brief Writes detected objects to a file.
 *
 * \process This process writes the detected objecs in the set to a
 * file. The actual renderingwriting is done by the selected \b
 * detected_object_set_output algorithm implementation.
 *
 * \iports
 *
 * \iport{image_file_name} Optional name of an image file to associate
 * with the set of detections.
 *
 * \iport{detected_object_set} Set ob objects to pass to writer
 * algorithm.
 *
 * \configs
 *
 * \config{file_name} Name of the file that the detections are written.
 *
 * \config{writer} Name of the configuration subblock that selects
 * and configures the writing algorithm
 */

//----------------------------------------------------------------
// Private implementation class
class detected_object_output_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;

  algo::detected_object_set_output_sptr m_writer;
}; // end priv class

// ================================================================

detected_object_output_process
::detected_object_output_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new detected_object_output_process::priv )
{
  make_ports();
  make_config();
}

detected_object_output_process
::~detected_object_output_process()
{
}

// ----------------------------------------------------------------
void detected_object_output_process
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
  if ( ! algo::detected_object_set_output::check_nested_algo_configuration_using_trait(
         writer,
         algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  algo::detected_object_set_output::set_nested_algo_configuration_using_trait(
    writer,
    algo_config,
    d->m_writer);
  if ( ! d->m_writer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
             "Unable to create writer." );
  }
}

// ----------------------------------------------------------------
void detected_object_output_process
::_init()
{
  scoped_init_instrumentation();

  d->m_writer->open( d->m_file_name ); // throws
}

// ----------------------------------------------------------------
void detected_object_output_process
::_step()
{
  std::string file_name;

  // image name is optional
  if ( has_input_port_edge_using_trait( image_file_name ) )
  {
    file_name = grab_from_port_using_trait( image_file_name );
  }

  kwiver::vital::detected_object_set_sptr input = grab_from_port_using_trait( detected_object_set );

  {
    scoped_step_instrumentation();

    d->m_writer->write_set( input, file_name );
  }
}

// ----------------------------------------------------------------
void detected_object_output_process
::_finalize()
{
  d->m_writer->complete();
}

// ----------------------------------------------------------------
void detected_object_output_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( image_file_name, optional );
  declare_input_port_using_trait( detected_object_set, required );
}

// ----------------------------------------------------------------
void detected_object_output_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( writer );
}

// ================================================================
detected_object_output_process::priv
::priv()
{
}

detected_object_output_process::priv
::~priv()
{
}

} // end namespace
