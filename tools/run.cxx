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
#include <set>
#include <filesystem>
#include <random>
#include <regex>

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
bool
option_takes_value( const std::string& arg )
{
  const std::set<std::string> flags =
    {"-h", "--help", "-D", "--dump-pipe", "--debug", "--no-reset-prompt",
     "--build-index", "--mosaic", "--gt-only", "--recursive"};
  return !arg.empty() && arg[0] == '-' && !flags.count(arg) &&
    arg.find('=') == std::string::npos &&
    !(arg.size() > 2 && (arg[1] == 's' || arg[1] == 'c' || arg[1] == 'I' || arg[1] == 'S'));
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

  bool positional_only = false;
  for( size_t i = 1; i < args.size(); ++i )
  {
    const auto& arg = args[i];
    if( arg == "--" && !positional_only ) { positional_only = true; continue; }
    if( !positional_only && !arg.empty() && arg[0] == '-' )
    {
      if( option_takes_value(arg) ) { ++i; }
      continue;
    }
    if( !arg.empty() && (positional_only || arg.find('=') == std::string::npos) )
    {
      found.push_back(arg);
    }
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
/// Stages live in a temporary directory; include paths and relativepath
/// directives retain the original pipeline's directory as their base.
int
run_staged_pipeline(
  const std::vector< std::string >& stages,
  const std::string& pipe_file_path,
  const std::vector< std::string >& applet_args )
{
  std::vector< std::string > forwarded_args;

  for( size_t i = 1; i < applet_args.size(); ++i )
  {
    const auto& arg = applet_args[i];
    if( arg == pipe_file_path || arg == "--" ) { continue; }
    forwarded_args.push_back(arg);
    if( option_takes_value(arg) && i + 1 < applet_args.size() )
    {
      forwarded_args.push_back(applet_args[++i]);
    }
  }

  namespace fs = std::filesystem;
  const auto pipe_dir = fs::absolute(pipe_file_path).parent_path();
  std::random_device random;
  fs::path temporary_dir;
  for( unsigned attempt = 0; attempt < 100; ++attempt )
  {
    auto candidate = fs::temp_directory_path() /
      ("viame-stage-" + std::to_string(random()) + "-" + std::to_string(random()));
    if( fs::create_directory(candidate) ) { temporary_dir = candidate; break; }
  }
  if( temporary_dir.empty() ) { throw std::runtime_error("Cannot create stage directory"); }
  struct cleanup
  {
    fs::path path;
    ~cleanup() { std::error_code ignored; fs::remove_all(path, ignored); }
  } guard{temporary_dir};

  std::cout << "Running staged pipeline with "
            << stages.size() << " stage(s)" << std::endl;

  for( size_t i = 0; i < stages.size(); ++i )
  {
    std::cout << std::endl << "=== Pipeline stage " << ( i + 1 )
              << " of " << stages.size() << " ===" << std::endl;

    const std::string tmp_path = (temporary_dir / (std::to_string(i + 1) + ".pipe")).string();

    {
      std::ofstream ofs( tmp_path );

      if( !ofs.is_open() )
      {
        std::cerr << "viame: Unable to write temporary pipe file: "
                  << tmp_path << std::endl;
        return EXIT_FAILURE;
      }

      // Relativepath values in the stage belong to the original pipe.
      // Includes get the original directory as their first search path.
      const std::regex relative(R"(^(\s*)relativepath\s+(\S+\s*=\s*)(.*)$)");
      std::istringstream lines(stages[i]);
      std::string line;
      while( std::getline(lines, line) )
      {
        std::smatch match;
        if( std::regex_match(line, match, relative) )
        {
          std::string value = match[3];
          const auto comment = value.find(" #");
          const std::string suffix = comment == std::string::npos ? "" : value.substr(comment);
          if( comment != std::string::npos ) { value.resize(comment); }
          const auto end = value.find_last_not_of(" \t");
          value = end == std::string::npos ? "" : value.substr(0, end + 1);
          ofs << match[1] << match[2] << (pipe_dir / value).lexically_normal().generic_string() << suffix << "\n";
        }
        else { ofs << line << "\n"; }
      }
      ofs.close();
      if( !ofs ) { throw std::runtime_error("Could not write stage pipeline"); }
    }

    std::vector< std::string > stage_args;
    stage_args.push_back( applet_args[0] );
    stage_args.push_back( tmp_path );
    stage_args.push_back("-I");
    stage_args.push_back(pipe_dir.string());
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
    if( wants_help(args) ) { return run_pipeline(args); }
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
