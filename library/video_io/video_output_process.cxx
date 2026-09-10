// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "video_output_process.h"

#include <viame/core_types/vital_types.h>
#include <viame/core_types/timestamp.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/image.h>
#include <viame/algorithm_framework/util/string.h>

#include <viame/algorithm_framework/algo/video_output.h>
#include <viame/algorithm_framework/exceptions.h>

#include <viame/pipeline_framework/type_traits.h>

#include <viame/pipeline_framework/process_exception.h>
#include <viame/pipeline_framework/datum.h>

#include <viame/core_types/video_settings.h>

#include <algorithm>
#include <cstdio>
#include <string>

namespace algo = kwiver::vital::algo;

namespace kwiver {

// (config-key, value-type, default-value, description )
create_config_trait( video_filename, std::string, "",
  "Name of output video file." );

create_config_trait( exit_on_invalid, bool, "false",
  "If a frame in the middle of a sequence is invalid, do not exit and throw "
  "an error, continue processing data. If the first frame cannot be read, "
  "always exit regardless of this setting." );

create_config_trait( maximum_length, std::string, "",
  "Maximum output video length (in seconds or HH:MM:SS format). If this "
  "length is exceeded, multiple video files less than this amount will be "
  "output with a timestamp start extension." );

create_config_trait( append_timestamp, bool, "false",
  "If true, append a timestamp to the filename when splitting video based on "
  "maximum_length." );

create_algorithm_name_config_trait( video_writer );

// -----------------------------------------------------------------------------
// Private implementation class
class video_output_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string                            m_video_filename;
  bool                                   m_exit_on_invalid;

  kwiver::vital::algo::video_output_sptr m_video_writer;
  kwiver::vital::algorithm_capabilities  m_video_traits;
  double                                 m_maximum_length;
  bool                                   m_append_timestamp;

  double                                 m_frame_rate;
  bool                                   m_is_first_frame;
  double                                 m_clip_start_time;
  kwiver::vital::image_container_sptr    m_last_frame;
  kwiver::vital::metadata_vector         m_last_metadata;

}; // end priv class


// =============================================================================
video_output_process
::video_output_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new video_output_process::priv )
{
  make_ports();
  make_config();
}

video_output_process
::~video_output_process()
{
  if( d->m_video_writer )
  {
    d->m_video_writer->close();
  }
}


// -----------------------------------------------------------------------------
void video_output_process
::_configure()
{
  scoped_configure_instrumentation();

  // Examine the configuration
  d->m_video_filename = config_value_using_trait( video_filename );
  d->m_exit_on_invalid = config_value_using_trait( exit_on_invalid );
  d->m_append_timestamp = config_value_using_trait( append_timestamp );

  std::string maximum_length_str = config_value_using_trait( maximum_length );

  if( !maximum_length_str.empty() )
  {
    // Parse time string (supports seconds or HH:MM:SS format)
    if( maximum_length_str.find( ':' ) != std::string::npos )
    {
      // Parse HH:MM:SS format
      int h = 0, m = 0, s = 0;
      if( sscanf( maximum_length_str.c_str(), "%d:%d:%d", &h, &m, &s ) >= 2 )
      {
        d->m_maximum_length = h * 3600.0 + m * 60.0 + s;
      }
    }
    else
    {
      d->m_maximum_length = std::stod( maximum_length_str );
    }
  }

  vital::config_block_sptr algo_config = get_config();

  if( !check_nested_algo_configuration_using_trait( video_writer, algo_config, d->m_video_writer ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Configuration check failed." );
  }

  // instantiate requested/configured algo type
  set_nested_algo_configuration_using_trait( video_writer, algo_config, d->m_video_writer );

  if( !d->m_video_writer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Unable to create video_writer." );
  }
}


// -----------------------------------------------------------------------------
// Post connection initialization
void video_output_process
::_init()
{
  scoped_init_instrumentation();

  d->m_is_first_frame = true;
  d->m_clip_start_time = -1.0;
}


// -----------------------------------------------------------------------------
void video_output_process
::_step()
{
  bool reset = false;

  vital::image_container_sptr frame = grab_from_port_using_trait( image );
  vital::timestamp ts = grab_from_port_using_trait( timestamp );

  if( !frame )
  {
    if( d->m_exit_on_invalid )
    {
      VITAL_THROW( vital::image_exception, "Invalid image received" );
    }
    else
    {
      frame = d->m_last_frame;
    }
  }
  else
  {
    d->m_last_frame = frame;
  }

  if( d->m_is_first_frame && has_input_port_edge_using_trait( frame_rate ) )
  {
    d->m_frame_rate = grab_from_port_using_trait( frame_rate );
  }

  if( d->m_maximum_length > 0.0 )
  {
    double current_seconds = ts.get_time_seconds();
    double clip_start_time = d->m_maximum_length *
      static_cast< int >( current_seconds / d->m_maximum_length );

    if( clip_start_time != d->m_clip_start_time )
    {
      d->m_clip_start_time = clip_start_time;
      reset = true;
    }
  }

  if( d->m_is_first_frame || reset )
  {
    // The writer needs the geometry and the frame rate before it can open,
    // and this is the only place that knows them: the geometry from the
    // frame in hand, the rate from the port. simple_video_settings carries
    // exactly those three, so this does not depend on any one arrow.
    vital::simple_video_settings default_settings(
      frame->width(), frame->height(), d->m_frame_rate );
    vital::video_settings const* settings = &default_settings;
    std::string filename = d->m_video_filename;

    if( reset && d->m_append_timestamp )
    {
      std::size_t pos = filename.find_last_of( "." );
      std::string stem = filename.substr( 0, pos );
      std::string ext = ( pos == std::string::npos ? "mp4" : filename.substr( pos ) );

      unsigned seconds_int = static_cast< unsigned >( d->m_clip_start_time );
      std::string h = std::to_string( seconds_int / 3600 );
      std::string m = std::to_string( ( seconds_int / 60 ) % 60 );
      std::string s = std::to_string( seconds_int % 60 );

      std::string time_str = "_" +
        std::string( 2 - std::min( std::size_t( 2 ), h.length() ), '0' ) + h + "h" +
        std::string( 2 - std::min( std::size_t( 2 ), m.length() ), '0' ) + m + "m" +
        std::string( 2 - std::min( std::size_t( 2 ), s.length() ), '0' ) + s + "s";

      filename = stem + time_str + ext;
    }

    d->m_video_writer->open( filename, settings ); // throws
    d->m_is_first_frame = false;
  }

  d->m_video_writer->add_image( frame, ts );

  if( has_input_port_edge_using_trait( metadata ) )
  {
    for( auto meta : grab_from_port_using_trait( metadata ) )
    {
      if( meta )
      {
        d->m_video_writer->add_metadata( *meta );
      }
    }
  }
}


// -----------------------------------------------------------------------------
// Flush and finalize the output video once the input stream ends. This hook is
// invoked when a complete datum arrives on all required input ports, after
// which _step() is never called again. Without it, the writer is only closed
// in the process destructor, which the scheduler may not invoke before the
// program exits -- leaving an MP4 with no moov atom and unflushed buffered
// packets, i.e. an unplayable file.
void video_output_process
::_finalize()
{
  scoped_finalize_instrumentation();

  if( d->m_video_writer )
  {
    d->m_video_writer->close();
  }
}


// -----------------------------------------------------------------------------
void video_output_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;

  // We are outputting a shared ref to the output image, therefore we
  // should mark it as shared.
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( metadata, optional );
  declare_input_port_using_trait( frame_rate, optional );
}


// -----------------------------------------------------------------------------
void video_output_process
::make_config()
{
  declare_config_using_trait( video_filename );
  declare_config_using_trait( exit_on_invalid );
  declare_config_using_trait( video_writer );
  declare_config_using_trait( maximum_length );
  declare_config_using_trait( append_timestamp );
}


// =============================================================================
video_output_process::priv
::priv()
  : m_exit_on_invalid( true ),
    m_maximum_length( 0.0 ),
    m_append_timestamp( false ),
    m_is_first_frame( true ),
    m_clip_start_time( -1.0 )
{
}

video_output_process::priv
::~priv()
{
}

} // end namespace
