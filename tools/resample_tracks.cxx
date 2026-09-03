/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Applet for resampling object tracks between frame rates

#include "resample_tracks.h"

#include <utilities_tracks.h>
#include <read_object_track_set_auto.h>
#include <write_object_track_set_viame_csv.h>

#include <kwiversys/SystemTools.hxx>

#include <vital/config/config_block.h>
#include <vital/logger/logger.h>
#include <vital/plugin_management/plugin_manager.h>
#include <vital/types/object_track_set.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace kv = kwiver::vital;

namespace viame {
namespace tools {

namespace {

// ----------------------------------------------------------------------------
size_t
count_states( const kv::object_track_set_sptr& tracks )
{
  size_t count = 0;

  for( const auto& trk : tracks->tracks() )
  {
    if( trk )
    {
      count += trk->size();
    }
  }

  return count;
}

} // namespace

// ----------------------------------------------------------------------------
void
resample_tracks_applet
::add_command_options()
{
  m_cmd_options->add_options()
    ( "h,help", "Display usage information",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "i,input", "Input track file (VIAME CSV, DIVE or COCO JSON)",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "o,output", "Output track file (VIAME CSV)",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "input-rate", "Frame rate (Hz) the input tracks were annotated at",
      ::cxxopts::value< double >()->default_value( "0" ), "value" )
    ( "output-rate", "Desired output frame rate (Hz), may be higher or lower "
      "than input",
      ::cxxopts::value< double >()->default_value( "0" ), "value" )
    ( "no-interpolate", "Copy the nearest annotated state instead of "
      "interpolating boxes",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "max-gap", "Maximum gap between annotated states, in input frames, to "
      "fill with new states (default: 0 = unlimited)",
      ::cxxopts::value< int >()->default_value( "0" ), "value" )
    ;
}

// ----------------------------------------------------------------------------
int
resample_tracks_applet
::run()
{
  kv::logger_handle_t logger = kv::get_logger( "viame.tools.resample_tracks" );

  auto& cmd_args = command_args();

  if( cmd_args[ "help" ].as< bool >() )
  {
    std::cout << "Usage: viame resample-tracks [options]\n\n"
              << "Resample object tracks from one video frame rate to another.\n"
              << "Frame numbers are rescaled to the output rate; states missing\n"
              << "at the new rate are filled by interpolating between annotated\n"
              << "states. Track extents are never extrapolated.\n"
              << m_cmd_options->help()
              << "\nExamples:\n"
              << "  viame resample-tracks -i tracks_5hz.csv -o tracks_10hz.csv"
              << " --input-rate 5 --output-rate 10\n"
              << "  viame resample-tracks -i tracks_30hz.csv -o tracks_5hz.csv"
              << " --input-rate 30 --output-rate 5\n"
              << std::endl;
    return EXIT_SUCCESS;
  }

  const std::string opt_input = cmd_args[ "input" ].as< std::string >();
  const std::string opt_output = cmd_args[ "output" ].as< std::string >();
  const double opt_input_rate = cmd_args[ "input-rate" ].as< double >();
  const double opt_output_rate = cmd_args[ "output-rate" ].as< double >();
  const bool opt_no_interpolate = cmd_args[ "no-interpolate" ].as< bool >();
  const int opt_max_gap = cmd_args[ "max-gap" ].as< int >();

  if( opt_input.empty() || opt_output.empty() )
  {
    LOG_ERROR( logger, "Both --input and --output must be specified" );
    return EXIT_FAILURE;
  }

  if( opt_input_rate <= 0.0 || opt_output_rate <= 0.0 )
  {
    LOG_ERROR( logger, "Both --input-rate and --output-rate must be "
                       "specified and positive" );
    return EXIT_FAILURE;
  }

  if( opt_max_gap < 0 )
  {
    LOG_ERROR( logger, "--max-gap cannot be negative" );
    return EXIT_FAILURE;
  }

  if( !kwiversys::SystemTools::FileExists( opt_input ) )
  {
    LOG_ERROR( logger, "Input file does not exist: " << opt_input );
    return EXIT_FAILURE;
  }

  // Needed for format-delegating readers
  kv::plugin_manager::instance().load_all_plugins();

  std::vector< kv::track_sptr > input_tracks;

  try
  {
    viame::read_object_track_set_auto reader;

    // Batch mode returns the full track set once; the default streaming mode
    // yields per-frame sets forever, relying on a pipeline to terminate it
    kv::config_block_sptr reader_config = reader.get_configuration();
    reader_config->set_value( "viame_csv:batch_load", "true" );
    reader_config->set_value( "dive:batch_load", "true" );
    reader.set_configuration( reader_config );

    reader.open( opt_input );

    kv::object_track_set_sptr track_set;

    while( reader.read_set( track_set ) )
    {
      if( !track_set )
      {
        continue;
      }

      for( const auto& trk : track_set->tracks() )
      {
        input_tracks.push_back( trk );
      }
    }

    reader.close();
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( logger, "Failed to read " << opt_input << ": " << e.what() );
    return EXIT_FAILURE;
  }

  kv::object_track_set_sptr input =
    std::make_shared< kv::object_track_set >( input_tracks );

  LOG_INFO( logger, "Read " << input->size() << " track(s) with "
            << count_states( input ) << " state(s) from " << opt_input );

  kv::object_track_set_sptr output = viame::resample_object_tracks(
    input,
    opt_input_rate,
    opt_output_rate,
    !opt_no_interpolate,
    static_cast< unsigned >( opt_max_gap ) );

  try
  {
    viame::write_object_track_set_viame_csv writer;

    std::ostringstream fps;
    fps << opt_output_rate;

    kv::config_block_sptr writer_config = kv::config_block::empty_config();
    writer_config->set_value( "frame_rate", fps.str() );

    writer.set_configuration( writer_config );
    writer.open( opt_output );
    writer.write_set( output, kv::timestamp(), "" );
    writer.close();
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( logger, "Failed to write " << opt_output << ": " << e.what() );
    return EXIT_FAILURE;
  }

  LOG_INFO( logger, "Wrote " << output->size() << " track(s) with "
            << count_states( output ) << " state(s) to " << opt_output );

  return EXIT_SUCCESS;
}

} // namespace tools
} // namespace viame
