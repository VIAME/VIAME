/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "run.h"

#include <vital/applets/applet_context.h>
#include <vital/config/config_block.h>
#include <vital/plugin_management/plugin_manager.h>

#ifdef VIAME_TOOLS_ENABLE_PYTHON
#include <python_script_applet.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace viame {
namespace tools {

namespace {

using applet_factory =
  kwiver::vital::implementation_factory_by_name< kwiver::tools::kwiver_applet >;

// ----------------------------------------------------------------------------
bool
ends_with( const std::string& str, const std::string& suffix )
{
  return str.size() >= suffix.size() &&
         str.compare( str.size() - suffix.size(), suffix.size(), suffix ) == 0;
}

// ----------------------------------------------------------------------------
/// Positional arguments of a command line whose element zero is the program.
///
/// A value handed to a flag does not count, and neither does a key=value
/// setting.
std::vector< std::string >
positional_args( const std::vector< std::string >& args )
{
  std::vector< std::string > found;

  for( size_t i = 1; i < args.size(); ++i )
  {
    const std::string& arg = args[i];

    if( arg.empty() || arg[0] == '-' || arg.find( '=' ) != std::string::npos )
    {
      continue;
    }

    if( !args[i - 1].empty() && args[i - 1][0] == '-' )
    {
      continue;
    }

    found.push_back( arg );
  }

  return found;
}

// ----------------------------------------------------------------------------
/// The pipe file to execute directly, or empty for batch processing.
///
/// The pipeline runner's only positional argument is the pipe file, so a lone
/// pipe file is a request for the runner. A pipe file with a companion input
/// (video, image, image list or folder) is the batch driver's shorthand form.
std::string
lone_pipe_file( const std::vector< std::string >& args )
{
  const auto positional = positional_args( args );

  if( positional.size() == 1 && ends_with( positional[0], ".pipe" ) )
  {
    return positional[0];
  }

  return {};
}

// ----------------------------------------------------------------------------
bool
wants_help( const std::vector< std::string >& args )
{
  for( size_t i = 1; i < args.size(); ++i )
  {
    if( args[i] == "-h" || args[i] == "--help" )
    {
      return true;
    }
  }

  return false;
}

// ----------------------------------------------------------------------------
/// The batch driver's own help only covers itself.
void
print_run_modes()
{
  std::cout
    << "viame run works in two modes, chosen by how you name the pipeline:"
    << std::endl << std::endl
    << "  viame run <pipeline.pipe> [options]" << std::endl
    << "      Execute one pipeline file directly. See \"viame help runner\""
    << std::endl
    << "      for the options that mode accepts." << std::endl << std::endl
    << "  viame run <pipeline> <video|image|image-list.txt|folder> [options]"
    << std::endl
    << "  viame run -d <directory> -p <pipeline.pipe> [options]" << std::endl
    << "      Process a video, image, image list or folder in batch, using"
    << std::endl
    << "      the options listed below. The pipeline may be a bare name"
    << std::endl
    << "      from configs/pipelines, e.g. detector_generic." << std::endl
    << std::endl
    << "  viame run <model.pt|.pth|.ckpt|.weights|.onnx|.zip> [<input>]"
    << std::endl
    << "      Wrap a bare model file in the default detector pipeline, or a"
    << std::endl
    << "      classifier in the frame classifier pipeline, and process the"
    << std::endl
    << "      input with it. RF-DETR, Ultralytics, MIT YOLO, LitDet,"
    << std::endl
    << "      MMDetection, Detectron2 and Darknet weights are recognized,"
    << std::endl
    << "      alone or zipped with their config files, as are netharn"
    << std::endl
    << "      deployed detectors and classifiers, ONNX files and packages,"
    << std::endl
    << "      and zips holding .pipe files (asking which to run when there"
    << std::endl
    << "      are several)."
    << std::endl
    << "      Without an input, only reports what the file was recognized as."
    << std::endl << std::endl;
}

// ----------------------------------------------------------------------------
/// Execute the pipeline runner applet on a full command line.
///
/// Element zero of args is the program name; the pipe file and the runner's
/// own options follow.
int
run_pipeline( std::vector< std::string > args )
{
  applet_factory app_fact;
  kwiver::tools::kwiver_applet_sptr applet(
    app_fact.create( "runner", kwiver::vital::config_block::empty_config() ) );

  kwiver::tools::applet_context context;
  context.m_applet_name = "runner";
  context.m_argv = args;
  // Same help wrapping as the tool runner gives every applet
  context.m_wtb.set_indent_string( "      " );

  applet->initialize( &context );
  applet->add_command_options();

  std::vector< char* > argv( args.size() + 1, nullptr );

  for( size_t i = 0; i < args.size(); ++i )
  {
    argv[i] = &args[i][0];
  }

  int argc = static_cast< int >( args.size() );
  char** argv_ptr = argv.data();

  cxxopts::ParseResult result = applet->m_cmd_options->parse( argc, argv_ptr );
  context.m_result = &result;

  return applet->run();
}

// ----------------------------------------------------------------------------
/// Split a pipe file on "pipeline stage N:" markers.
///
/// Returns the per-stage pipeline text in stage order, or nothing when the
/// file has no markers or they are not numbered sequentially from 1.
std::vector< std::string >
scan_for_stages( const std::string& pipe_file_path )
{
  std::ifstream ifs( pipe_file_path );

  if( !ifs.is_open() )
  {
    return {};
  }

  struct stage_entry
  {
    int number;
    std::string content;
  };

  std::vector< stage_entry > entries;
  std::string line;
  int current_stage = -1;
  std::ostringstream current_content;

  while( std::getline( ifs, line ) )
  {
    std::string trimmed = line;
    size_t start = trimmed.find_first_not_of( " \t" );

    if( start != std::string::npos )
    {
      trimmed = trimmed.substr( start );
    }

    const std::string prefix = "pipeline stage ";

    if( trimmed.compare( 0, prefix.size(), prefix ) == 0 )
    {
      std::string rest = trimmed.substr( prefix.size() );
      size_t colon = rest.find( ':' );

      if( colon != std::string::npos )
      {
        std::string num_str = rest.substr( 0, colon );

        size_t ns = num_str.find_first_not_of( " \t" );
        size_t ne = num_str.find_last_not_of( " \t" );

        if( ns != std::string::npos )
        {
          num_str = num_str.substr( ns, ne - ns + 1 );
        }

        try
        {
          int stage_num = std::stoi( num_str );

          if( current_stage > 0 )
          {
            entries.push_back( { current_stage, current_content.str() } );
            current_content.str( "" );
            current_content.clear();
          }

          current_stage = stage_num;
          continue;
        }
        catch( ... )
        {
          // Not a stage marker after all, keep it as pipeline text
        }
      }
    }

    if( current_stage > 0 )
    {
      current_content << line << "\n";
    }
  }

  if( current_stage > 0 )
  {
    entries.push_back( { current_stage, current_content.str() } );
  }

  if( entries.empty() )
  {
    return {};
  }

  std::sort( entries.begin(), entries.end(),
    []( const stage_entry& a, const stage_entry& b )
    {
      return a.number < b.number;
    } );

  std::vector< std::string > result;

  for( size_t i = 0; i < entries.size(); ++i )
  {
    if( entries[i].number != static_cast< int >( i + 1 ) )
    {
      std::cerr << "viame: Pipeline stages must be numbered sequentially "
                << "starting from 1.  Found stage " << entries[i].number
                << " at position " << ( i + 1 ) << "." << std::endl;
      return {};
    }

    result.push_back( entries[i].content );
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Run each stage as its own pipeline, in order, stopping at the first
/// failure. Settings (-s), config files (-c) and include paths (-I) from the
/// command line apply to every stage.
///
/// Each stage is written to a temporary pipe file beside the original so
/// that include and relativepath directives resolve the same way.
int
run_staged_pipeline(
  const std::vector< std::string >& stages,
  const std::string& pipe_file_path,
  const std::vector< std::string >& applet_args )
{
  std::vector< std::string > forwarded_args;

  for( size_t i = 0; i < applet_args.size(); ++i )
  {
    const std::string& arg = applet_args[i];

    if( ( arg == "-s" || arg == "--setting" ||
          arg == "-c" || arg == "--config"  ||
          arg == "-I" || arg == "--include" ) && i + 1 < applet_args.size() )
    {
      forwarded_args.push_back( arg );
      forwarded_args.push_back( applet_args[++i] );
    }
    else if( arg.compare( 0, 2, "-s" ) == 0 && arg.size() > 2
             && arg[2] != '-' )
    {
      forwarded_args.push_back( arg );
    }
  }

  std::string pipe_dir;
  {
    size_t slash = pipe_file_path.find_last_of( "/\\" );

    if( slash != std::string::npos )
    {
      pipe_dir = pipe_file_path.substr( 0, slash + 1 );
    }
  }

  const auto pid = getpid();

  std::cout << "Running staged pipeline with "
            << stages.size() << " stage(s)" << std::endl;

  for( size_t i = 0; i < stages.size(); ++i )
  {
    std::cout << std::endl << "=== Pipeline stage " << ( i + 1 )
              << " of " << stages.size() << " ===" << std::endl;

    std::ostringstream tmp_name;
    tmp_name << pipe_dir << ".viame_stage_" << ( i + 1 )
             << "_" << pid << ".pipe";
    const std::string tmp_path = tmp_name.str();

    {
      std::ofstream ofs( tmp_path );

      if( !ofs.is_open() )
      {
        std::cerr << "viame: Unable to write temporary pipe file: "
                  << tmp_path << std::endl;
        return EXIT_FAILURE;
      }

      ofs << stages[i];
    }

    std::vector< std::string > stage_args;
    stage_args.push_back( applet_args[0] );
    stage_args.push_back( tmp_path );
    stage_args.insert( stage_args.end(),
                       forwarded_args.begin(), forwarded_args.end() );

    int result = EXIT_FAILURE;

    try
    {
      result = run_pipeline( stage_args );
    }
    catch( const std::exception& e )
    {
      std::cerr << "viame: Stage " << ( i + 1 )
                << " failed: " << e.what() << std::endl;
      std::remove( tmp_path.c_str() );
      return EXIT_FAILURE;
    }
    catch( ... )
    {
      std::cerr << "viame: Stage " << ( i + 1 )
                << " failed with unknown error." << std::endl;
      std::remove( tmp_path.c_str() );
      return EXIT_FAILURE;
    }

    std::remove( tmp_path.c_str() );

    if( result != EXIT_SUCCESS )
    {
      std::cerr << "viame: Stage " << ( i + 1 )
                << " exited with error code " << result << std::endl;
      return result;
    }

    std::cout << "Stage " << ( i + 1 ) << " completed successfully."
              << std::endl;
  }

  std::cout << std::endl << "All " << stages.size()
            << " pipeline stage(s) completed successfully." << std::endl;

  return EXIT_SUCCESS;
}

#ifdef VIAME_TOOLS_ENABLE_PYTHON

VIAME_PYTHON_SCRIPT_APPLET( run_bulk_applet, "run-bulk", "run_bulk.py",
  "Process videos or images in batch" )

#endif

} // namespace

// ----------------------------------------------------------------------------
int
run_applet
::run()
{
  const auto& args = applet_args();

  const std::string pipe_file = lone_pipe_file( args );

  if( !pipe_file.empty() )
  {
    const auto stages = scan_for_stages( pipe_file );

    if( !stages.empty() )
    {
      return run_staged_pipeline( stages, pipe_file, args );
    }

    return run_pipeline( args );
  }

  if( args.size() < 2 || wants_help( args ) )
  {
    print_run_modes();
  }

#ifdef VIAME_TOOLS_ENABLE_PYTHON
  kwiver::tools::applet_context context;
  context.m_applet_name = applet_name();
  context.m_argv = args;

  run_bulk_applet bulk;
  bulk.initialize( &context );

  return bulk.run();
#else
  std::cerr << "viame run: batch processing needs a python-enabled build. "
            << "Give a single pipe file to execute it directly." << std::endl;
  return EXIT_FAILURE;
#endif
}

} // namespace tools
} // namespace viame
