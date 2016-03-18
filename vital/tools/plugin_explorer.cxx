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

#include <vital/algorithm_plugin_manager.h>
#include <vital/algo/algorithm.h>
#include <vital/registrar.h>
#include <vital/config/config_block.h>
#include <vital/util/tokenize.h>
#include <vital/vital_foreach.h>
#include <kwiversys/CommandLineArguments.hxx>

#include <iostream>
#include <vector>
#include <string>

// Global options
bool opt_config( false );
bool opt_detail( false );
bool opt_help( false );
std::vector< std::string > opt_path;


// ------------------------------------------------------------------
void
print_config( kwiver::vital::config_block_sptr config )
{
  kwiver::vital::config_block_keys_t all_keys = config->available_values();
  std::string indent( "    " );

  std::cout << indent << "Configuration block contents\n";

  VITAL_FOREACH( kwiver::vital::config_block_key_t key, all_keys )
  {
    kwiver::vital::config_block_value_t val = config->get_value< kwiver::vital::config_block_value_t > ( key );
    std::cout << std::endl
              << indent << "\"" << key << "\" = \"" << val << "\"\n";

    kwiver::vital::config_block_description_t descr = config->get_description( key );
    std::cout << indent << "Description: " << descr << std::endl;
  }
}


// ------------------------------------------------------------------
void
detailed_algo()
{
  const std::vector< std::string > reg_names =  kwiver::vital::algorithm::registered_names();

  VITAL_FOREACH( std::string const& name, reg_names )
  {
    std::vector< std::string > token;
    kwiver::vital::tokenize( name, token, ":" );

    // create instance type_name : impl_name
    kwiver::vital::algorithm_sptr ptr = kwiver::vital::algorithm::create( token[0], token[1] );

    std::cout << "\n---------------------"
              << "\nDetailed info on type \"" << token[0] << "\" implementation \"" << token[1] << "\""
              << std::endl;

//     if ( opt_config )
    {
      // Get configuration
      kwiver::vital::config_block_sptr config = ptr->get_configuration();

      // print config -- (optional)
      print_config( config );
    }
  }
}


// ------------------------------------------------------------------
void
print_help()
{
  std::cout << "This program loads Map-TK plugins and displays their data.\n"
            << "Additional paths can be specified in \"KWIVER_PLUGIN_PATH\" environment variable\n"
            << "\n"
            << "Options are:\n"
            << "  --help           displays usage information\n"
            << "  --plugin-name    load only these plugins\n"
            << "  --path name      also load plugins from this directory (can appear multiple times)\n"
            << "  -Iname           also load plugins from this directory (can appear multiple times)\n"
            << "  --detail  -d     generate detailed listing\n"
            << "  --config         display plugin configuration\n"
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
  arg.AddArgument( "--plugin-name", argT::SPACE_ARGUMENT, &plugin_name, "Display usage information" );
  arg.AddArgument( "--detail",      argT::NO_ARGUMENT, &opt_detail, "Display detailed information about plugins" );
  arg.AddArgument( "-d",            argT::NO_ARGUMENT, &opt_detail, "Display detailed information about plugins" );
  arg.AddArgument( "--config",      argT::NO_ARGUMENT, &opt_config, "Display configuration information needed by plugins" );
  arg.AddCallback( "--path",        argT::SPACE_ARGUMENT, path_callback, 0, "Add directory to plugin search path" );
  arg.AddCallback( "-I",            argT::CONCAT_ARGUMENT, path_callback, 0, "Add directory to plugin search path" );

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

  kwiver::vital::algorithm_plugin_manager& apm = kwiver::vital::algorithm_plugin_manager::instance();

  VITAL_FOREACH( std::string const& path, opt_path )
  {
    apm.add_search_path( path );
  }

  // locate all plugins
  apm.register_plugins( plugin_name );

  std::string path_string;
  auto search_path( apm.get_search_path() );
  VITAL_FOREACH( auto module_dir, search_path )
  {
    path_string += module_dir + ":";
  }

  std::cout << "---- Algorithm search path\n"
            << path_string << std::endl
            << std::endl;

  std::cout << "---- Registered module names:\n";
  std::vector< std::string > module_list = apm.registered_module_names();
  VITAL_FOREACH( std::string const& name, module_list )
  {
    std::cout << "   " << name << std::endl;
  }

  std::cout << "\n---- registered algorithms (type_name:impl_name)\n";
  VITAL_FOREACH( std::string const& name, kwiver::vital::algorithm::registered_names() )
  {
    std::cout << "   " << name << std::endl;
  }

  // currently this is the same as --config option
  if ( opt_detail )
  {
    detailed_algo();
  }

  // Need a way to introspect these modules

#if 0
  kwver::vital::registrar& reg = kwiver::vital::registrar::instance();

  std::vector< std::string > reg_list = reg.registered_names< XXX > ();
  std::cout << "\n\n---- Resigtered algorithm names\n";
  VITAL_FOREACH( std::string const& name, reg_list )
  {
    std::cout << "    " << name << std::endl;
  }
#endif

  return 0;
} // main
