/*ckwg +29
 * Copyright 2016-2018 by Kitware, Inc.
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

/// \file
///
/// This program reads a video and extracts all the KLV metadata.

#include <iostream>
#include <fstream>

#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/exceptions.h>
#include <vital/util/get_paths.h>

#include <vital/algo/video_input.h>

#include <vital/types/metadata.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <kwiversys/CommandLineArguments.hxx>

// Global options
bool        opt_help( false );
std::string opt_config;         // config file name
std::string opt_out_config;     // output config file name

typedef kwiversys::CommandLineArguments argT;

//+ maybe not needed
// ------------------------------------------------------------------
static kwiver::vital::config_block_sptr default_config()
{
  kwiver::vital::config_block_sptr config =
    kwiver::vital::config_block::empty_config( "dump_klv_tool" );

  config->set_value( "video_reader:type", "vidl_ffmpeg",
                     "Implementation for video reader." );
  config->set_value( "video_reader:vidl_ffmpeg:time_source",  "misp",
                     "Time source for reader." );

  kwiver::vital::algo::video_input::get_nested_algo_configuration(
    "video_reader", config, kwiver::vital::algo::video_input_sptr() );

  return config;
}

// ----------------------------------------------------------------
/** Main entry.
 *
 *
 */
int main( int argc, char** argv )
{
  kwiversys::CommandLineArguments arg;

  arg.Initialize( argc, argv );
  arg.StoreUnusedArguments( true );

  arg.AddArgument( "--help",        argT::NO_ARGUMENT, &opt_help, "Display usage information" );
  arg.AddArgument( "-h",              argT::NO_ARGUMENT, &opt_help, "Display usage information" );
  arg.AddArgument( "--config",      argT::SPACE_ARGUMENT, &opt_config, "Configuration file for tool" );
  arg.AddArgument( "-c",            argT::SPACE_ARGUMENT, &opt_config, "Configuration file for tool" );
  arg.AddArgument( "--output-config", argT::SPACE_ARGUMENT, &opt_out_config, "Dump configuration for tool" );

  if ( ! arg.Parse() )
  {
    std::cerr << "Problem parsing arguments" << std::endl;
    return EXIT_FAILURE;
  }

  if ( opt_help )
  {
    std::cerr
      << "USAGE: " << argv[0] << " [OPTS]  video-file\n\n"
      << "Options:"
      << arg.GetHelp() << std::endl;
    return EXIT_SUCCESS;
  }

  char** newArgv = 0;
  int newArgc = 0;
  arg.GetUnusedArguments(&newArgc, &newArgv);

  if( newArgc == 1 )
  {
    std::cout << "Missing file name.\n"
      << "Usage: " << newArgv[0] << " video-file-name\n" << std::endl;

      return EXIT_FAILURE;
  }

  std::string video_file = newArgv[1];

  arg.DeleteRemainingArguments(newArgc, &newArgv);

  // register the algorithm implementations
  std::string rel_plugin_path = kwiver::vital::get_executable_path() + "/../lib/modules";
  kwiver::vital::plugin_manager::instance().add_search_path(rel_plugin_path);
  kwiver::vital::plugin_manager::instance().load_all_plugins();
  kwiver::vital::algo::video_input_sptr video_reader;
  kwiver::vital::config_block_sptr config = default_config();

  // If --config given, read in config file, merge in with default just generated
  if( ! opt_config.empty() )
  {
    config->merge_config( kwiver::vital::read_config_file( opt_config ) );
  }

  kwiver::vital::algo::video_input::set_nested_algo_configuration( "video_reader", config, video_reader );
  kwiver::vital::algo::video_input::get_nested_algo_configuration( "video_reader", config, video_reader );
  // Check to see if we are to dump config
  if ( ! opt_out_config.empty() )
  {
    std::ofstream fout( opt_out_config.c_str() );
    if( ! fout )
    {
      std::cout << "Couldn't open \"" << opt_out_config << "\" for writing.\n";
      return EXIT_FAILURE;
    }

    config->print( fout );
    std::cout << "Wrote config to \"" << opt_out_config << "\". Exiting.\n";
    return EXIT_SUCCESS;
  }

  if( !kwiver::vital::algo::video_input::check_nested_algo_configuration( "video_reader", config ) )
  {
    std::cerr << "Invalid video_reader config" << std::endl;
    return EXIT_FAILURE;
  }

  // instantiate a video reader
  try
  {
    video_reader->open( video_file );
  }
  catch ( kwiver::vital::video_exception const& e )
  {
    std::cerr << "Video Exception-Couldn't open " << video_file << std::endl
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  catch ( kwiver::vital::file_not_found_exception const& e )
  {
    std::cerr << "Couldn't open " << video_file << std::endl
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  kwiver::vital::algorithm_capabilities const& caps = video_reader->get_implementation_capabilities();
  if ( ! caps.capability( kwiver::vital::algo::video_input::HAS_METADATA ) )
  {
    std::cerr << "No metadata stream found in " << video_file << '\n';
    return EXIT_FAILURE;
  }

  int count(1);
  kwiver::vital::image_container_sptr frame;
  kwiver::vital::timestamp ts;

  while ( video_reader->next_frame( ts ) )
  {
    std::cout << "========== Read frame " << ts.get_frame()
              << " (index " << count << ") ==========" << std::endl;

    kwiver::vital::metadata_vector metadata = video_reader->frame_metadata();
    for( auto meta : metadata )
    {
      std::cout << "\n\n---------------- Metadata from: " << meta->timestamp() << std::endl;
      print_metadata( std::cout, *meta );
      ++count;
    }

  } // end while

  std::cout << "-- End of video --\n";

  return EXIT_SUCCESS;
}
