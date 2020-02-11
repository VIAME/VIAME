/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <vital/applets/kwiver_applet.h>
#include <vital/applets/applet_context.h>

#include <vital/applets/applet_registrar.h>
#include <vital/exceptions/base.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/plugin_loader/plugin_filter_category.h>
#include <vital/plugin_loader/plugin_filter_default.h>
#include <vital/plugin_loader/plugin_manager.h>
#include <vital/util/get_paths.h>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <exception>
#include <memory>

using applet_factory = kwiver::vital::implementation_factory_by_name< kwiver::tools::kwiver_applet >;
using applet_context_t = std::shared_ptr< kwiver::tools::applet_context >;

// ============================================================================
/**
 * This class processes the incoming list of command line options.
 * They are separated into oprtions for the tool runner and options
 * for the applet.
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

    // The first applet args is the program name.
    m_applet_args.push_back( "kwiver" );

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
          // found applet name
          m_applet_name = std::string( (*p_argv)[i] );
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
  std::string m_output_file; // empty is no file specified

  std::vector<std::string> m_runner_args;
  std::vector<std::string> m_applet_args;
  std::string m_applet_name;
};


// ----------------------------------------------------------------------------
/**
 * Generate list of all applets that have been discovered.
 */
void tool_runner_usage( applet_context_t ctxt,
                        kwiver::vital::plugin_manager& vpm )
{
  // display help message
  std::cout << "Usage: kwiver  <applet>  [args]" << std::endl
            << "<applet> can be one of the following:" << std::endl
            << "help - prints this message" << std::endl;

  // Get list of factories for implementations of the applet
  const auto fact_list = vpm.get_factories( typeid( kwiver::tools::kwiver_applet ).name() );

  // Loop over all factories in the list and display name and description
  for( auto fact : fact_list )
  {
    std::string buf = "-- Not Set --";
    fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, buf );
    std::cout << "    " << buf << " - ";

    fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, buf );

    // All we want is the first line of the description.
    size_t pos = buf.find_first_of('\n');
    if ( pos != 0 )
    {
      // Take all but the ending newline
      buf = buf.substr( 0, pos );
    }

    std::cout << ctxt->m_wtb.wrap_text( buf );
  }
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
int main(int argc, char *argv[])
{
  //
  // Global shared context
  // Allocated on the stack so it will automatically clean up
  //
  applet_context_t tool_context = std::make_shared< kwiver::tools::applet_context >();

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  const std::string exec_path = kwiver::vital::get_executable_path();
  vpm.add_search_path(exec_path + "/../lib/kwiver/plugins");

  // remove all default plugin filters
  vpm.get_loader()->clear_filters();

  // Add filter to select all plugins
  kwiver::vital::plugin_filter_handle_t filt = std::make_shared<kwiver::vital::plugin_filter_default>();
  vpm.get_loader()->add_filter( filt );

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
    std::cerr << "Command argument error: " << e.what() << std::endl;
    exit( -1 );
  }
  catch ( kwiver::vital::plugin_factory_not_found& )
  {
    std::cerr << "Tool \"" << argv[1] << "\" not found. Type \""
              << argv[0] << " help\" to list available tools." << std::endl;

    exit(-1);
  }
  catch ( kwiver::vital::vital_exception& e )
  {
    std::cerr << "Caught unhandled kwiver::vital::vital_exception: " << e.what() << std::endl;
    exit( -1 );
  }
  catch ( std::exception& e )
  {
    std::cerr << "Caught unhandled std::exception: " << e.what() << std::endl;
    exit( -1 );
  }
  catch ( ... )
  {
    std::cerr << "Caught unhandled exception" << std::endl;
    exit( -1 );
  }

  return 0;
}
