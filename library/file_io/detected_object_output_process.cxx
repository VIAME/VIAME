// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for detected_object_set_output process
 */

#include "detected_object_output_process.h"

#include <viame/core_types/vital_types.h>
#include <viame/algorithm_framework/exceptions.h>
#include <viame/core_types/timestamp.h>
#include <viame/algorithm_framework/algo/detected_object_set_output.h>

#include <viame/pipeline_framework/type_traits.h>

#include <viame/pipeline_framework/process_exception.h>

#include <fstream>
#include <memory>
#include <ctime>

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( file_name, std::string, "",
  "Name of the detection set file to write." );
create_config_trait( frame_list_output, std::string, "",
  "Optional frame list output to also write." );
create_config_trait( write_time_as_uid, bool, "false",
  "Always write the frame time (HH:MM:SS.ssssss) as the identifier passed to "
  "the writer, even when an image file name is available. Regardless of this "
  "setting the frame time is used whenever there is no image file name, which "
  "is the case for video, so the identifier is a time for video and a file "
  "name for image lists. Requires the timestamp port to be connected." );

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

// -----------------------------------------------------------------------------
// Private implementation class
class detected_object_output_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_file_name;
  std::string m_frame_list_output;
  bool m_write_time_as_uid{ false };

  /// Frame time as HH:MM:SS.ssssss, empty if it cannot be formatted.
  std::string format_time( kwiver::vital::timestamp const& ts ) const;

  algo::detected_object_set_output_sptr m_writer;
  std::unique_ptr< std::ofstream > m_frame_list_writer;
}; // end priv class

// ================================================================

detected_object_output_process
::detected_object_output_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new detected_object_output_process::priv )
{
  // Required so that we can do 1 step past the end
  set_data_checking_level( check_none );

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
  d->m_frame_list_output = config_value_using_trait( frame_list_output );
  d->m_write_time_as_uid = config_value_using_trait( write_time_as_uid );

  if( d->m_file_name.empty() )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
             "Required file name not specified." );
  }

  std::size_t time_pos = d->m_file_name.find( "[CURRENT_TIME]" );
  if( time_pos != std::string::npos )
  {
    char buffer[256];
    time_t raw;
    struct tm *t;
    time( &raw );
    t = localtime( &raw );

    strftime( buffer, sizeof( buffer ), "%Y%m%d_%H%M%S", t );
    d->m_file_name.replace( time_pos, 14, buffer );

    std::size_t frame_time_pos = d->m_frame_list_output.find( "[CURRENT_TIME]" );
    if( !d->m_frame_list_output.empty() && frame_time_pos != std::string::npos )
    {
      d->m_frame_list_output.replace( frame_time_pos, 14, buffer );
    }
  }

  if( !d->m_frame_list_output.empty() )
  {
    d->m_frame_list_writer.reset( new std::ofstream( d->m_frame_list_output ) );
  }

  // Get algo config entries
  kwiver::vital::config_block_sptr algo_config = get_config(); // config for process

  // validate configuration
  if ( ! check_nested_algo_configuration_using_trait(
         writer,
         algo_config, d->m_writer ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Configuration check failed." );
  }

  // instantiate image reader and converter based on config type
  set_nested_algo_configuration_using_trait(
    writer,
    algo_config,
    d->m_writer);
  if ( !d->m_writer )
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
  auto datum = peek_at_datum_using_trait( detected_object_set );

  if ( datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( detected_object_set );
    mark_process_as_complete();

    d->m_writer->complete();

    return;
  }

  std::string file_name;
  std::string identifier;

  // image name is optional. The frame list records the file name as given,
  // while the identifier handed to the writer may be replaced by the frame
  // time below, so the two are tracked separately.
  if ( has_input_port_edge_using_trait( image_file_name ) )
  {
    file_name = grab_from_port_using_trait( image_file_name );
    identifier = file_name;
  }

  // timestamp is optional, but must always be consumed when connected,
  // otherwise the edge backs up and stalls the pipeline
  if ( has_input_port_edge_using_trait( timestamp ) )
  {
    auto const ts = grab_from_port_using_trait( timestamp );

    // Video has no per-frame file name, so fall back to the frame time there
    // and leave image lists writing their file names
    if ( ( d->m_write_time_as_uid || identifier.empty() ) &&
         ts.has_valid_time() )
    {
      identifier = d->format_time( ts );
    }
  }

  if ( d->m_frame_list_writer )
  {
    *d->m_frame_list_writer << file_name << std::endl;
  }

  kwiver::vital::detected_object_set_sptr input =
    grab_from_port_using_trait( detected_object_set );

  {
    scoped_step_instrumentation();

    d->m_writer->write_set( input, identifier );
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
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( detected_object_set, required );
}

// ----------------------------------------------------------------
void detected_object_output_process
::make_config()
{
  declare_config_using_trait( file_name );
  declare_config_using_trait( frame_list_output );
  declare_config_using_trait( write_time_as_uid );
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
  if( m_frame_list_writer )
  {
    m_frame_list_writer->close();
  }
}


// ----------------------------------------------------------------
std::string
detected_object_output_process::priv
::format_time( kwiver::vital::timestamp const& ts ) const
{
  const kwiver::vital::time_usec_t usec( 1000000 );
  const std::time_t time_s =
    static_cast< std::time_t >( ts.get_time_usec() / usec );
  const unsigned time_us =
    static_cast< unsigned >( ts.get_time_usec() % usec );

  char buffer[10];
  struct tm* tmp = gmtime( &time_s );

  if( !tmp || !strftime( buffer, sizeof( buffer ), "%H:%M:%S", tmp ) )
  {
    return std::string();
  }

  std::string time_us_str = std::to_string( time_us );
  while( time_us_str.size() < 6 )
  {
    time_us_str = "0" + time_us_str;
  }

  return std::string( buffer ) + "." + time_us_str;
}

} // end namespace
