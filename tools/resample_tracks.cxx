/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Command-line tool for resampling object tracks between frame rates

#include <utilities_tracks.h>
#include <read_object_track_set_auto.h>
#include <write_object_track_set_viame_csv.h>

#include <kwiversys/CommandLineArguments.hxx>
#include <kwiversys/SystemTools.hxx>

#include <vital/config/config_block.h>
#include <vital/logger/logger.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/types/object_track_set.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace kv = kwiver::vital;

class resample_tracks_params
{
public:
  kwiversys::CommandLineArguments m_args;

  bool opt_help = false;

  std::string opt_input;
  std::string opt_output;

  double opt_input_rate = 0.0;
  double opt_output_rate = 0.0;

  bool opt_no_interpolate = false;
  int opt_max_gap = 0;
};

static resample_tracks_params g_params;
static kv::logger_handle_t g_logger;


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


int main( int argc, char* argv[] )
{
  g_logger = kv::get_logger( "viame.tools.resample_tracks" );

  typedef kwiversys::CommandLineArguments argT;

  g_params.m_args.Initialize( argc, argv );

  g_params.m_args.AddArgument( "--help", argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );
  g_params.m_args.AddArgument( "-h", argT::NO_ARGUMENT,
    &g_params.opt_help, "Display usage information" );

  g_params.m_args.AddArgument( "--input", argT::SPACE_ARGUMENT,
    &g_params.opt_input, "Input track file (VIAME CSV, DIVE or COCO JSON)" );
  g_params.m_args.AddArgument( "-i", argT::SPACE_ARGUMENT,
    &g_params.opt_input, "Input track file (VIAME CSV, DIVE or COCO JSON)" );
  g_params.m_args.AddArgument( "--output", argT::SPACE_ARGUMENT,
    &g_params.opt_output, "Output track file (VIAME CSV)" );
  g_params.m_args.AddArgument( "-o", argT::SPACE_ARGUMENT,
    &g_params.opt_output, "Output track file (VIAME CSV)" );

  g_params.m_args.AddArgument( "--input-rate", argT::SPACE_ARGUMENT,
    &g_params.opt_input_rate,
    "Frame rate (Hz) the input tracks were annotated at" );
  g_params.m_args.AddArgument( "--output-rate", argT::SPACE_ARGUMENT,
    &g_params.opt_output_rate,
    "Desired output frame rate (Hz), may be higher or lower than input" );

  g_params.m_args.AddArgument( "--no-interpolate", argT::NO_ARGUMENT,
    &g_params.opt_no_interpolate,
    "Copy the nearest annotated state instead of interpolating boxes" );
  g_params.m_args.AddArgument( "--max-gap", argT::SPACE_ARGUMENT,
    &g_params.opt_max_gap,
    "Maximum gap between annotated states, in input frames, to fill with "
    "new states (default: 0 = unlimited)" );

  if( !g_params.m_args.Parse() )
  {
    LOG_ERROR( g_logger, "Problem parsing arguments" );
    return EXIT_FAILURE;
  }

  if( g_params.opt_help )
  {
    std::cout << "Usage: " << argv[0] << " [options]\n\n"
              << "Resample object tracks from one video frame rate to another.\n"
              << "Frame numbers are rescaled to the output rate; states missing\n"
              << "at the new rate are filled by interpolating between annotated\n"
              << "states. Track extents are never extrapolated.\n\n"
              << "Options:\n"
              << g_params.m_args.GetHelp()
              << "\nExamples:\n"
              << "  " << argv[0] << " -i tracks_5hz.csv -o tracks_10hz.csv"
              << " --input-rate 5 --output-rate 10\n"
              << "  " << argv[0] << " -i tracks_30hz.csv -o tracks_5hz.csv"
              << " --input-rate 30 --output-rate 5\n"
              << std::endl;
    return EXIT_SUCCESS;
  }

  if( g_params.opt_input.empty() || g_params.opt_output.empty() )
  {
    LOG_ERROR( g_logger, "Both --input and --output must be specified" );
    return EXIT_FAILURE;
  }

  if( g_params.opt_input_rate <= 0.0 || g_params.opt_output_rate <= 0.0 )
  {
    LOG_ERROR( g_logger, "Both --input-rate and --output-rate must be "
                         "specified and positive" );
    return EXIT_FAILURE;
  }

  if( g_params.opt_max_gap < 0 )
  {
    LOG_ERROR( g_logger, "--max-gap cannot be negative" );
    return EXIT_FAILURE;
  }

  if( !kwiversys::SystemTools::FileExists( g_params.opt_input ) )
  {
    LOG_ERROR( g_logger, "Input file does not exist: " << g_params.opt_input );
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

    reader.open( g_params.opt_input );

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
    LOG_ERROR( g_logger, "Failed to read " << g_params.opt_input
               << ": " << e.what() );
    return EXIT_FAILURE;
  }

  kv::object_track_set_sptr input =
    std::make_shared< kv::object_track_set >( input_tracks );

  LOG_INFO( g_logger, "Read " << input->size() << " track(s) with "
            << count_states( input ) << " state(s) from "
            << g_params.opt_input );

  kv::object_track_set_sptr output = viame::resample_object_tracks(
    input,
    g_params.opt_input_rate,
    g_params.opt_output_rate,
    !g_params.opt_no_interpolate,
    static_cast< unsigned >( g_params.opt_max_gap ) );

  try
  {
    viame::write_object_track_set_viame_csv writer;

    std::ostringstream fps;
    fps << g_params.opt_output_rate;

    kv::config_block_sptr writer_config = kv::config_block::empty_config();
    writer_config->set_value( "frame_rate", fps.str() );

    writer.set_configuration( writer_config );
    writer.open( g_params.opt_output );
    writer.write_set( output, kv::timestamp(), "" );
    writer.close();
  }
  catch( const std::exception& e )
  {
    LOG_ERROR( g_logger, "Failed to write " << g_params.opt_output
               << ": " << e.what() );
    return EXIT_FAILURE;
  }

  LOG_INFO( g_logger, "Wrote " << output->size() << " track(s) with "
            << count_states( output ) << " state(s) to "
            << g_params.opt_output );

  return EXIT_SUCCESS;
}
