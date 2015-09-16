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

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>

#include <iostream>

namespace po = boost::program_options;

// Global options
bool opt_config(false);



// ------------------------------------------------------------------
void print_config( kwiver::vital::config_block_sptr config )
{
  kwiver::vital::config_block_keys_t all_keys = config->available_values();
  std::string indent( "    " );

  std::cout << indent<< "Configuration block contents\n";

  for ( kwiver::vital::config_block_key_t key : all_keys )
  {
    kwiver::vital::config_block_value_t val = config->get_value< kwiver::vital::config_block_value_t >( key );
    std::cout << std::endl
              << indent << "\"" << key << "\" = \"" << val << "\"\n";

    kwiver::vital::config_block_description_t descr = config->get_description( key );
    std::cout << indent << "Description: " << descr << std::endl;
  }
}


// ------------------------------------------------------------------
void detailed_algo()
{
  std::vector< std::string > reg_names =  kwiver::vital::algorithm::registered_names();
  for ( std::string const& name : reg_names )
  {
    std::vector< std::string > token;
    boost::split( token, name, boost::is_any_of( ":" ) );

    // create instance type_name : impl_name
    kwiver::vital::algorithm_sptr ptr = kwiver::vital::algorithm::create( token[0], token[1] );

    std::cout << "\n---------------------"
              << "\nDetailed info on type \"" << token[0] << "\" implementation \"" << token[1] << "\""
              << std::endl;

    // Get configuration
    kwiver::vital::config_block_sptr config = ptr->get_configuration();

    if ( opt_config )
    {
      // print config -- (optional)
      print_config( config );
    }
  }
}


// ------------------------------------------------------------------
int main(int argc, char *argv[])
{
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("plugin-name", po::value< std::string >(), "load only these plugins")
    ("path,I", po::value< std::vector< std::string> >(), "add directory to search")
    ("config", "Display algorithm configuration block")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << "This program loads and displays MapTK algorithms.\n"
              << "Additional paths can be specified in \"VITAL_PLUGIN_PATH\" environment variable\n";

    std::cout << desc << "\n";
    return 1;
  }

  if (vm.count("config"))
  {
    opt_config = true;
  }

  kwiver::vital::algorithm_plugin_manager& apm = kwiver::vital::algorithm_plugin_manager::instance();

  //
  // Add any commane line specified path components
  //
  if (vm.count( "path" ) )
  {
    for ( std::string const& p : vm["path"].as<std::vector< std::string > >() )
    {
      apm.add_search_path( p );
    }
  }

  //
  // Use selected plugin name if supplied
  //
  std::string plugin_name;
  if (vm.count( "plugin-name" ) )
  {
    plugin_name = vm["plugin_name"].as<std::string>();
  }

  // locate all plugins
  apm.register_plugins( plugin_name );

  std::cout << "---- Algorithm search path\n"
            << apm.get_search_path()
            << std::endl << std::endl;

  std::cout << "---- Registered module names:\n";
  std::vector< std::string >module_list = apm.registered_module_names();
  for ( std::string const& name : module_list)
  {
    std::cout << "   " << name << std::endl;
  }

  std::cout << "\n---- registered algorithms (type_name:impl_name)\n";
  for ( std::string const& name : kwiver::vital::algorithm::registered_names())
  {
    std::cout << "   " << name << std::endl;
  }

  // currently this is the same as --config option
  detailed_algo();

  // Need a way to introspect these modules

#if 0
  kwver::vital::registrar& reg = kwiver::vital::registrar::instance();

  std::vector< std::string > reg_list = reg.registered_names< XXX >();
  std::cout << "\n\n---- Resigtered algorithm names\n";
  for ( std::string const& name : reg_list)
  {
    std::cout << "    " << name << std::endl;
  }
#endif

  return 0;
}
