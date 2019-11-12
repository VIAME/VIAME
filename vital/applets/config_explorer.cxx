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

#include "config_explorer.h"

#include <vital/plugin_loader/plugin_manager.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block.h>
#include <vital/config/config_parser.h>

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

namespace kwiver {
namespace tools {

namespace {


} // end namespace


// ----------------------------------------------------------------------------
void
config_explorer::
add_command_options()
{
  m_cmd_options->custom_help( wrap_text( "[options] config-file\n"
    "This program assists in debugging config loading problems. It loads a "
    "configuration and displays the contents or displays the search path."
    "Additional paths can be specified in \"KWIVER_CONFIG_PATH\" environment variable"
    "or on the command line with the -I or --path options."
    "If -ds is specified, the detailed search paths that apply to the application are"
    "displayed only, otherwise the config file is loaded."
    "\n\n"
    "The option -dc only has effect when a config file is specified and causes a"
    "detailed output of the config entries."
    "\n\n"
    "If -I or --path are specified, then the config file is only searched for using"
    "the specified path. The application name based paths are not used."
                                ) );

 m_cmd_options->positional_help( "\n  config-file  - name of configuration file." );

  m_cmd_options->add_options()
    ( "h,help", "Display applet usage" )
    ( "ds", "Display detailed application search path" )
    ( "dc", "Display detailed config contents" )
    ( "I,path", "Add directory to config search path", cxxopts::value<std::vector<std::string>>() )
    ( "a,application", "Application name", cxxopts::value<std::string>() )
    ( "v,version", "Application version", cxxopts::value<std::string>() )
    ( "p,prefix", "Non-standard installation prefix. (e.g. /opt/kitware)", cxxopts::value<std::string>() )

    // positional parameters
    ( "config-file", "configuration file", cxxopts::value<std::string>())
    ;

  m_cmd_options->parse_positional("config-file");
}

// ============================================================================
config_explorer::
config_explorer()
{ }


// ----------------------------------------------------------------------------
int
config_explorer::
run()
{
  std::string opt_app_name = applet_name();
  auto& cmd_args = command_args();

  if ( cmd_args["help"].as<bool>() )
  {
    std::cout << m_cmd_options->help();
    return EXIT_SUCCESS;
  }

  bool opt_detail_ds = cmd_args["ds"].as<bool>();
  bool opt_detail_dc = cmd_args["dc"].as<bool>();

  // Note that you have to determine that the options is there before getting value
  // Seg-fault otherwise
  std::string opt_app_version = cmd_args.count("version") ? cmd_args["version"].as<std::string>() : "";
  std::string opt_install_prefix = cmd_args.count("prefix") ? cmd_args["prefix"].as<std::string>() : "";
  std::vector< std::string > opt_path;

  if ( cmd_args.count("path") > 0 )
  {
    opt_path = cmd_args["path"].as<std::vector<std::string>>();
  }

  if ( cmd_args.count("application") )
  {
    opt_app_name = cmd_args["application"].as<std::string>();
  }

  // Display application specific search path.
  if ( opt_detail_ds )
  {
    // test for invalid option combination
    if ( opt_detail_dc
         || cmd_args.count("application")
         || cmd_args.count("version")
         || cmd_args.count("prefix")
      )
    {
      std::cerr << "Invalid set of options specified with --ds\n";
      return EXIT_FAILURE;
    }

    kwiver::vital::config_path_list_t search_path =
      kwiver::vital::application_config_file_paths( opt_app_name,
                                                    opt_app_version,
                                                    opt_install_prefix );

    std::cout << "Application specific configuration search paths for\n"
              << "       App name: " << opt_app_name << std::endl
              << "    App version: " << opt_app_version << std::endl
              << " Install Prefix: " << opt_install_prefix << std::endl
              << std::endl;

    for( auto path : search_path )
    {
      std::cout << path << std::endl;
    }

    return EXIT_SUCCESS;
  }

  std::string opt_config_file;

  if ( cmd_args.count("config-file") )
  {
    opt_config_file = cmd_args["config-file"].as<std::string>();
  }
  else
  {
    std::cout << "Missing config-file name.\n"
      <<  m_cmd_options->help()
      << std::endl;

    return EXIT_FAILURE;
  }

  kwiver::vital::config_block_sptr config;

  if ( ! opt_path.empty() )
  {
    std::cout << "Using custom search path.\n";
    config = kwiver::vital::read_config_file( opt_config_file,
                                              opt_path );
  }
  else
  {
    std::cout << "Using application default search path.\n";
    config = kwiver::vital::read_config_file( opt_config_file,
                                              opt_app_name,
                                              opt_app_version,
                                              opt_install_prefix,
                                              true );  // merge all configs
  }

  //
  // Dump details of config
  //
  if ( opt_detail_dc )
  {
    std::cout << "Config contents for\n"
              << "       App name: " << opt_app_name << std::endl
              << "    App version: " << opt_app_version << std::endl
              << " Install Prefix: " << opt_install_prefix << std::endl
              << std::endl;

    kwiver::vital::write_config( config, std::cout );
  }

  return EXIT_SUCCESS;
} // run

} } // end namespace
