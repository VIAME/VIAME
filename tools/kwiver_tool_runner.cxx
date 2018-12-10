/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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


#include "kwiver_applet.h"
#include "applet_context.h"

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/exceptions/base.h>

#include <kwiversys/SystemTools.hxx>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <exception>
#include <memory>

using applet_factory = kwiver::vital::implementation_factory_by_name< kwiver::tools::kwiver_applet >;
using applet_context_t = std::shared_ptr< kwiver::tools::applet_context >;

// ============================================================================
class command_line_parser
{
public:
  command_line_parser( int p_argc, char *** p_argv )
  {
    int state(0);

    // Parse the command line
    // Command line format:
    // arg0 [runner-flags] <applet> [applet-flags]

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
      buf = buf.substr( 0, pos-1 );
    }

    std::cout << ctxt->m_wtb.wrap_text( buf );
  }
}


// ----------------------------------------------------------------------------
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
  kwiver::tools::kwiver_applet_sptr applet( app_fact.create( options.m_applet_args[1] ) );
  tool_context->m_applet_name = options.m_applet_args[1];
  applet->initialize( tool_context.get() );
  applet->usage( std::cout );
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
    applet->initialize( tool_context.get() );

    // Run the specified tool
    return applet->run( options.m_applet_args );
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
