/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/plugin_loader/plugin_factory.h>
#include <vital/config/config_block.h>
#include <vital/util/demangle.h>
#include <vital/vital_foreach.h>
#include <kwiversys/CommandLineArguments.hxx>
#include <kwiversys/RegularExpression.hxx>

#include <iostream>
#include <vector>
#include <string>

// Global options
bool opt_config( false );
bool opt_detail( false );
bool opt_help( false );
bool opt_path_list( false );
bool opt_brief( false );
bool opt_modules( false );
bool opt_files( false );
std::vector< std::string > opt_path;
std::string opt_fact_regex;


bool opt_fact_filt( false ); // internal option for factory filtering
kwiversys::RegularExpression fact_regex;


// ------------------------------------------------------------------
/*
 * Functor to print an attribute
 */
struct print_functor
{
  print_functor( std::ostream& str)
    : m_str( str )
  { }

  void operator() ( std::string const& key, std::string const& val ) const
  {
    // Skip canonical names and other attributes that are displayed elsewhere
    if ( ( kwiver::vital::plugin_factory::PLUGIN_NAME != key)
         && ( kwiver::vital::plugin_factory::CONCRETE_TYPE != key )
         && ( kwiver::vital::plugin_factory::INTERFACE_TYPE != key )
         && ( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION != key )
         && ( kwiver::vital::plugin_factory::PLUGIN_FILE_NAME != key )
         && ( kwiver::vital::plugin_factory::PLUGIN_VERSION != key )
      )
    {
      std::string value(val);

      size_t pos = value.find( '\004' );
      if ( std::string::npos != pos)
      {
        value.replace( pos, 1, "\"  ::  \"" );
      }

      m_str << "    * " << key << ": \"" << value << "\"" << std::endl;
    }
  }

  std::ostream& m_str;
};


// ------------------------------------------------------------------
void
print_help()
{
  std::cout << "This program loads vital plugins and displays their data.\n"
            << "Additional paths can be specified in \"KWIVER_PLUGIN_PATH\" environment variable\n"
            << "\n"
            << "Options are:\n"
            << "  --help           displays usage information\n"
            << "  --path           display plugin search path\n"
            << "  -Iname           also load plugins from this directory (can appear multiple times)\n"
            << "  --detail  -d     generate detailed listing\n"
            << "  --fact  regex    display factories that match regexp\n"
            << "  --brief          display factory name and description only\n"
            << "  --mod            display list of loaded modules\n"
            << "  --files          display list of files successfully opened to load plugins\n"
  ;

  return;
}


// ------------------------------------------------------------------
int
path_callback( const char*  argument,   // name of argument
               const char*  value,      // value of argument
               void*        call_data ) // data from register call
{
  const std::string p( value );

  opt_path.push_back( p );
  return 1;   // return true for OK
}


// ------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  std::string plugin_name;

  kwiversys::CommandLineArguments arg;

  arg.Initialize( argc, argv );
  typedef kwiversys::CommandLineArguments argT;

  arg.AddArgument( "--help",        argT::NO_ARGUMENT, &opt_help, "Display usage information" );
  arg.AddArgument( "-h",            argT::NO_ARGUMENT, &opt_help, "Display usage information" );
  arg.AddArgument( "--plugin-name", argT::SPACE_ARGUMENT, &plugin_name, "Optional name of single plugin to display." );
  arg.AddArgument( "--detail",      argT::NO_ARGUMENT, &opt_detail, "Display detailed information about plugins" );
  arg.AddArgument( "-d",            argT::NO_ARGUMENT, &opt_detail, "Display detailed information about plugins" );
  arg.AddArgument( "--config",      argT::NO_ARGUMENT, &opt_config, "Display configuration information needed by plugins" );
  arg.AddArgument( "--path",        argT::NO_ARGUMENT, &opt_path_list, "Display plugin search path" );
  arg.AddCallback( "-I",            argT::CONCAT_ARGUMENT, path_callback, 0, "Add directory to plugin search path" );
  arg.AddArgument( "--fact",        argT::SPACE_ARGUMENT, &opt_fact_regex, "Filter factories based on regexp" );
  arg.AddArgument( "--brief",       argT::NO_ARGUMENT, &opt_brief, "Brief display" );
  arg.AddArgument( "--files",       argT::NO_ARGUMENT, &opt_files, "Display list of loaded files" );
  arg.AddArgument( "--mod",         argT::NO_ARGUMENT, &opt_modules, "Display list of loaded modules" );

  if ( ! arg.Parse() )
  {
    std::cerr << "Problem parsing arguments" << std::endl;
    exit( 0 );
  }

  if ( opt_help )
  {
    print_help();
    exit( 0 );
  }

  // If a factory filtering regex specified, then compile it.
  if ( ! opt_fact_regex.empty() )
  {
    opt_fact_filt = true;
    if ( ! fact_regex.compile( opt_fact_regex) )
    {
      std::cerr << "Invalid regular expression \"" << opt_fact_regex << "\"" << std::endl;
      return 1;
    }
  }

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();

  VITAL_FOREACH( std::string const& path, opt_path )
  {
    vpm.add_search_path( path );
  }

  if ( opt_path_list )
  {
    std::cout << "---- Plugin search path\n";

    std::string path_string;
    std::vector< kwiver::vital::path_t > const search_path( vpm.search_path() );
    VITAL_FOREACH( auto module_dir, search_path )
    {
      std::cout << "    " << module_dir << std::endl;
    }
    std::cout << std::endl;
  }

  if ( opt_modules )
  {
    std::cout << "---- Registered module names:\n";
    auto module_list = vpm.module_map();
    VITAL_FOREACH( auto const name, module_list )
    {
      std::cout << "   " << name.first << "  loaded from: " << name.second << std::endl;
    }
    std::cout << std::endl;
  }

  //
  // display factory list
  //
  auto plugin_map = vpm.plugin_map();

  std::cout << "\n---- Registered Factories\n";
  VITAL_FOREACH( auto it, plugin_map )
  {
    std::string ds = kwiver::vital::demangle( it.first );

    // If regexp matching is enabled, and this does not match, skip it
    if ( opt_fact_filt && ( ! fact_regex.find( ds ) ) )
    {
      continue;
    }

    std::cout << "\nFactories that create type \"" << ds << "\"" << std::endl;

    // Get vector of factories
    kwiver::vital::plugin_factory_vector_t const & facts = it.second;

    VITAL_FOREACH( kwiver::vital::plugin_factory_handle_t const fact, facts )
    {
      // Print the required fields first
      std::string buf;
      buf = "-- Not Set --";
      fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, buf );

      std::string version("");
      fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, version );

      std::cout << "  Plugin name: " << buf;
      if ( ! version.empty() )
      {
        std::cout << "      Version: " << version << std::endl;
      }

      buf = "-- Not Set --";
      fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, buf );
      std::cout << "      Description: " << buf << std::endl;

      if ( opt_brief )
      {
        continue;
      }

      if ( opt_detail)
      {
        buf = "-- Not Set --";
        if ( fact->get_attribute( kwiver::vital::plugin_factory::CONCRETE_TYPE, buf ) )
        {
          buf = kwiver::vital::demangle( buf );
        }
        std::cout << "      Creates concrete type: " << buf << std::endl;
      }

      if ( opt_detail )
      {
        buf = "-- Not Set --";
        fact->get_attribute( kwiver::vital::plugin_factory::PLUGIN_FILE_NAME, buf );
        std::cout << "      Plugin loaded from file: " << buf << std::endl;

        // print all the rest of the attributes
        print_functor pf( std::cout );
        fact->for_each_attr( pf );
      }

      // Need to check for specific well known factory types.
      // 1) sprokit::process
      // 2) sprokit::scheduler
      // Have separate classes decode and format each factory type.
      // Need a way to get options from these delegates and accept selected options,
      //

      //
      // For processes, need to display:
      // option for "hidden" config entries and ports
      // Need to instantiate each process to get these details.
      // 1) process config
      // 2) input ports
      // 3) output ports

      //
      // Schedulers
      // This is not much different from the default plugin display.
      // Name and description come from attributes.
      // 1) name
      // 2) description

      //
      // Code to format these data can be lifted from processopedia.
      //

    } // end foreach factory
  } // end foreach interface type

  //
  // list files is specified
  //
  if ( opt_files )
  {
    std::vector< std::string > file_list = vpm.file_list();

    std::cout << "\n---- Files Successfully Opened" << std::endl;
    VITAL_FOREACH( std::string const& name, file_list )
    {
      std::cout << "  " << name << std::endl;
    } // end foreach
  }

  return 0;
} // main
