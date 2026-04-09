// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/**
 * \file viame.cxx
 *
 * \brief VIAME tool runner - main entry point for VIAME command line tools.
 *
 * This tool functions identically to the kwiver tool runner, allowing you
 * to run pipeline files and other applets via subcommands:
 *
 *   viame runner my_pipeline.pipe
 *   viame help
 *   viame explore-config my_config.conf
 *
 * As a convenience, if the first argument is a .pipe file, the runner
 * applet is automatically invoked:
 *
 *   viame my_pipeline.pipe          # equivalent to: viame runner my_pipeline.pipe
 *
 * Similarly, if the first argument is a .conf file, the train applet
 * is automatically invoked:
 *
 *   viame my_config.conf            # equivalent to: viame train -c my_config.conf
 *
 * It loads all VIAME and KWIVER plugins and dispatches to the appropriate
 * applet based on the subcommand name.
 */

#include <vital/applets/kwiver_applet.h>
#include <vital/applets/applet_context.h>

#include <vital/applets/applet_registrar.h>
#include <vital/exceptions/base.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/plugin_loader/plugin_filter_category.h>
#include <vital/plugin_loader/plugin_filter_default.h>
#include <vital/plugin_loader/plugin_manager_internal.h>
#include <vital/util/get_paths.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using applet_factory = kwiver::vital::implementation_factory_by_name< kwiver::tools::kwiver_applet >;
using applet_context_t = std::shared_ptr< kwiver::tools::applet_context >;

// ============================================================================
/**
 * Check if a string ends with a given suffix.
 */
static bool ends_with( const std::string& str, const std::string& suffix )
{
  if ( suffix.size() > str.size() )
  {
    return false;
  }
  return str.compare( str.size() - suffix.size(), suffix.size(), suffix ) == 0;
}

// ============================================================================
/**
 * This class processes the incoming list of command line options.
 * They are separated into options for the tool runner and options
 * for the applet.
 *
 * Special case: if the first non-flag argument looks like a pipeline file
 * (ends with .pipe), automatically use the "runner" applet and treat
 * the argument as the pipeline file path.
 *
 * Special case: if the first non-flag argument looks like a config file
 * (ends with .conf), automatically use the "train" applet with the
 * -c/--config option.
 */
class command_line_parser
{
public:
  command_line_parser( int p_argc, char *** p_argv )
  {
    int state(0);

    // Parse the command line
    // Command line format:
    // arg0 [runner-flags] <applet> [applet-args]
    // OR
    // arg0 [runner-flags] <pipeline.pipe> [applet-args]  (implicit runner)

    // The first applet args is the program name.
    m_applet_args.push_back( "viame" );

    // Separate args into groups
    for (int i = 1; i < p_argc; ++i )
    {
      if (state == 0)
      {
        // look for an option flag
        if ( (*p_argv)[i][0] == '-' )
        {
          m_runner_args.push_back( (*p_argv)[i] );
        }
        else
        {
          // found applet name (or pipeline file)
          std::string arg = std::string( (*p_argv)[i] );

          // Check if this looks like a pipeline file
          if ( ends_with( arg, ".pipe" ) )
          {
            // Implicit runner mode: treat as "runner <pipeline.pipe>"
            m_applet_name = "runner";
            m_applet_args.push_back( arg );
          }
          // Check if this looks like a config file
          else if ( ends_with( arg, ".conf" ) )
          {
            // Implicit train mode: treat as "train -c <config.conf>"
            m_applet_name = "train";
            m_applet_args.push_back( "-c" );
            m_applet_args.push_back( arg );
          }
          else
          {
            // Normal applet name
            m_applet_name = arg;
          }
          state = 1; // advance state
        }
      }
      else if (state == 1)
      {
        // Collecting applet parameters
        m_applet_args.push_back( (*p_argv)[i] );
      }
    } // end for

  }

  // ----------------------
  // tool runner arguments.
  std::string m_output_file; // empty if no file specified

  std::vector<std::string> m_runner_args;
  std::vector<std::string> m_applet_args;
  std::string m_applet_name;
};

// ----------------------------------------------------------------------------
/**
 * Generate list of all applets that have been discovered.
 */
void tool_runner_usage( VITAL_UNUSED applet_context_t ctxt,
                        kwiver::vital::plugin_manager& vpm )
{
  // display help message
  std::cout << "VIAME - Video and Image Analytics for Marine Environments" << std::endl
            << std::endl
            << "Usage: viame <applet> [args]" << std::endl
            << std::endl
            << "<applet> can be one of the following:" << std::endl
            << "  help - prints this message" << std::endl
            << std::endl
            << "Available applets are listed below:" << std::endl
            << std::endl;

  // Get list of factories for implementations of the applet
  const auto fact_list = vpm.get_factories( typeid( kwiver::tools::kwiver_applet ).name() );

  // Loop over all factories in the list and display name and description
  using help_pair = std::pair< std::string, std::string >;
  std::vector< help_pair > help_text;
  size_t tab_stop(0);

  for( auto fact : fact_list )
  {
    std::string buf = "-- Not Set --";
    fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, buf );

    std::string descr = "-- Not Set --";
    fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, descr );

    // All we want is the first line of the description.
    size_t pos = descr.find_first_of('\n');
    if ( pos != 0 )
    {
      // Take all but the ending newline
      descr = descr.substr( 0, pos );
    }

    help_text.push_back( help_pair({ buf, descr }) );
    tab_stop = std::max( tab_stop, buf.size() );
  } // end for

  // add some space after the longest applet name
  tab_stop += 2;

  // sort the applet names
  sort( help_text.begin(), help_text.end() );

  for ( auto const& elem : help_text )
  {
    const size_t filler = tab_stop - elem.first.size();
    std::cout << "  " << elem.first << std::string( filler, ' ') << elem.second << std::endl;
  }

  std::cout << std::endl
            << "Common examples:" << std::endl
            << "  viame my_pipeline.pipe              # Run a pipeline file (shorthand)" << std::endl
            << "  viame runner my_pipeline.pipe       # Run a pipeline file (explicit)" << std::endl
            << "  viame my_config.conf                # Train with a config file (shorthand)" << std::endl
            << "  viame train -c my_config.conf       # Train with a config file (explicit)" << std::endl
            << "  viame help runner                   # Get help on the runner applet" << std::endl
            << "  viame explore-config my.conf        # Explore configuration file" << std::endl
            << std::endl
            << "Note: If the first argument ends with .pipe, the runner applet is" << std::endl
            << "automatically invoked. So 'viame x.pipe' is equivalent to 'viame runner x.pipe'." << std::endl
            << "Similarly, if the first argument ends with .conf, the train applet is" << std::endl
            << "automatically invoked. So 'viame x.conf' is equivalent to 'viame train -c x.conf'." << std::endl
            << std::endl;
}

// ----------------------------------------------------------------------------
/**
 * This function handles the "help" operation. If there is an arg
 * after the help, then that arg is taken to be the applet name and
 * help is displayed for it.
 *
 * If the only arg is "help", then call the function to generate short
 * help for all known applets.
 */
void help_applet( const command_line_parser& options,
                  applet_context_t tool_context,
                  kwiver::vital::plugin_manager& vpm )
{
  if ( options.m_applet_args.size() < 2 )
  {
    tool_runner_usage( tool_context, vpm );
    return;
  }

  // Create applet based on the name provided
  applet_factory app_fact;
  std::string buf = "-- Not Set --";
  auto fact = app_fact.find_factory( options.m_applet_args[1] );
  fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, buf );

  kwiver::tools::kwiver_applet_sptr applet( app_fact.create( options.m_applet_args[1] ) );
  tool_context->m_applet_name = options.m_applet_args[1];
  applet->initialize( tool_context.get() );
  applet->add_command_options();

  // display help text
  std::cout << applet->m_cmd_options->help();
}

// ============================================================================
/**
 * Scan a .pipe file for "pipeline stage N:" markers.
 *
 * If markers are found, the file content is split into per-stage pipeline
 * text and returned as a vector (index 0 = stage 1, etc.).  If no markers
 * are found, an empty vector is returned, signalling normal single-pipeline
 * execution.
 */
static std::vector< std::string >
scan_for_stages( const std::string& pipe_file_path )
{
  std::ifstream ifs( pipe_file_path );

  if( !ifs.is_open() )
  {
    return {};
  }

  // First pass: collect (stage_number, content) pairs
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
    // Check for stage marker: "pipeline stage N:"
    // Allow leading whitespace.
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

        // Trim whitespace from number
        size_t ns = num_str.find_first_not_of( " \t" );
        size_t ne = num_str.find_last_not_of( " \t" );

        if( ns != std::string::npos )
        {
          num_str = num_str.substr( ns, ne - ns + 1 );
        }

        try
        {
          int stage_num = std::stoi( num_str );

          // Save previous stage
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
          // Not a valid stage marker, treat as normal content
        }
      }
    }

    if( current_stage > 0 )
    {
      current_content << line << "\n";
    }
  }

  // Save last stage
  if( current_stage > 0 )
  {
    entries.push_back( { current_stage, current_content.str() } );
  }

  if( entries.empty() )
  {
    return {};
  }

  // Sort by stage number and validate sequential numbering from 1
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

// ============================================================================
/**
 * Run a multi-stage pipeline.  Each stage is a complete pipeline definition
 * that is built, executed, and torn down before the next stage begins.
 * All command-line settings (-s), config files (-c), and include paths (-I)
 * are applied to every stage.
 *
 * Implementation: for each stage we write a temporary .pipe file and
 * dispatch to a fresh pipeline_runner applet instance via the plugin
 * factory.  This avoids depending on sprokit headers that are not
 * installed by kwiver.
 */
static int
run_staged_pipeline(
  const std::vector< std::string >& stages,
  const std::string& pipe_file_path,
  const std::vector< std::string >& applet_args )
{
  // Collect forwarded options from the command line (-s, -c, -I).
  // These are appended to the argument list for every stage.
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

  // Resolve the pipe file directory so that temporary stage files live
  // alongside the original (for correct relativepath resolution).
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

    // Write stage content to a temporary .pipe file next to the original
    // so that include/relativepath directives resolve correctly.
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

    // Build argument list for this stage
    std::vector< std::string > stage_args;
    stage_args.push_back( "viame" );    // program name
    stage_args.push_back( tmp_path );   // pipe file (positional)
    stage_args.insert( stage_args.end(),
                       forwarded_args.begin(), forwarded_args.end() );

    int result = EXIT_FAILURE;

    try
    {
      applet_factory app_fact;
      kwiver::tools::kwiver_applet_sptr applet( app_fact.create( "runner" ) );

      auto stage_context =
        std::make_shared< kwiver::tools::applet_context >();
      stage_context->m_applet_name = "runner";
      stage_context->m_argv = stage_args;

      applet->initialize( stage_context.get() );
      applet->add_command_options();

      // Convert args to argv style for cxxopts
      std::vector< char* > argv_vect( stage_args.size() + 1, nullptr );

      for( size_t a = 0; a < stage_args.size(); ++a )
      {
        argv_vect[a] = &stage_args[a][0];
      }

      int local_argc = static_cast< int >( stage_args.size() );
      char** local_argv = argv_vect.data();

      cxxopts::ParseResult local_result =
        applet->m_cmd_options->parse( local_argc, local_argv );
      stage_context->m_result = &local_result;

      result = applet->run();
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

    // Clean up temp file
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

// ============================================================================
int main(int argc, char *argv[])
{
  //
  // Global shared context
  // Allocated on the stack so it will automatically clean up
  //
  applet_context_t tool_context = std::make_shared< kwiver::tools::applet_context >();

  kwiver::vital::plugin_manager_internal& vpm = kwiver::vital::plugin_manager_internal::instance();

  // Add VIAME and KWIVER plugin search paths
  const std::string exec_path = kwiver::vital::get_executable_path();

  // VIAME plugin paths (relative to executable)
  vpm.add_search_path(exec_path + "/../lib/viame/plugins");
  vpm.add_search_path(exec_path + "/../lib/kwiver/plugins");

  // Also check standard locations
  vpm.add_search_path(exec_path + "/../lib/modules");
  vpm.add_search_path(exec_path + "/../lib/sprokit");

  // Load all available plugins
  vpm.load_all_plugins();

  // initialize the global context
  tool_context->m_wtb.set_indent_string( "      " );

  command_line_parser options( argc, &argv );

  if ( (options.m_applet_name == "help") || (argc == 1) )
  {
    help_applet( options, tool_context, vpm );
    return 0;
  } // end help code

  // ----------------------------------------------------------------------------
  // Check for staged pipeline before normal applet dispatch.
  // If the pipe file contains "pipeline stage N:" markers, run stages
  // sequentially instead of dispatching to the runner applet.
  if( options.m_applet_name == "runner" )
  {
    std::string pipe_file;

    for( const auto& arg : options.m_applet_args )
    {
      if( ends_with( arg, ".pipe" ) )
      {
        pipe_file = arg;
        break;
      }
    }

    if( !pipe_file.empty() )
    {
      auto stages = scan_for_stages( pipe_file );

      if( !stages.empty() )
      {
        return run_staged_pipeline( stages, pipe_file,
                                    options.m_applet_args );
      }
    }
  }

  // ----------------------------------------------------------------------------
  try
  {
    // Create applet based on the name provided
    applet_factory app_fact;
    kwiver::tools::kwiver_applet_sptr applet( app_fact.create( options.m_applet_name ) );

    tool_context->m_applet_name = options.m_applet_name;
    tool_context->m_argv = options.m_applet_args; // save a copy of the args

    // Pass the context to the applet. This is done as a separate call
    // because the default factory for applets does not take any
    // parameters.
    applet->initialize( tool_context.get() );

    // Call the applet so it can add the commands that it is looking
    // for.
    applet->add_command_options();

    int local_argc = 0;
    char** local_argv = 0;
    std::vector<char *> argv_vect;

    // There are some cases where the applet wants to do its own
    // command line parsing (e.g. QT apps). If this flag is not set,
    // then we will parse our standard arg set
    if ( ! tool_context->m_skip_command_args_parsing )
    {
      // Convert args list back to argv style. :-(
      // This is needed for the command options support package.
      argv_vect.resize( options.m_applet_args.size() +1, nullptr );
      for (std::size_t i = 0; i != options.m_applet_args.size(); ++i)
      {
        argv_vect[i] = &options.m_applet_args[i][0];
      }
    }
    else
    {
      argv_vect.resize( 2, nullptr );
      argv_vect[0] = &options.m_applet_args[0][0];
    }

    local_argc = argv_vect.size()-1;
    local_argv = &argv_vect[0];

    // The parse result has to be created locally due to class design.
    // No default CTOR, copy CTOR or operation.
    cxxopts::ParseResult local_result = applet->m_cmd_options->parse( local_argc, local_argv );

    // Make results available in the context,
    tool_context->m_result = &local_result; // in this case the address of a stack variable is o.k.

    // Run the specified tool
    return applet->run();
  }
  catch ( cxxopts::OptionException& e)
  {
    std::cerr << "viame: Command argument error: " << e.what() << std::endl;
    exit( -1 );
  }
  catch ( kwiver::vital::plugin_factory_not_found& )
  {
    std::cerr << "viame: Applet \"" << argv[1] << "\" not found." << std::endl
              << "Type \"viame help\" to list available applets." << std::endl;

    exit(-1);
  }
  catch ( kwiver::vital::vital_exception& e )
  {
    std::cerr << "viame: Caught unhandled kwiver::vital::vital_exception: " << e.what() << std::endl;
    exit( -1 );
  }
  catch ( std::exception& e )
  {
    std::cerr << "viame: Caught unhandled std::exception: " << e.what() << std::endl;
    exit( -1 );
  }
  catch ( ... )
  {
    std::cerr << "viame: Caught unhandled exception" << std::endl;
    exit( -1 );
  }

  return 0;
}
