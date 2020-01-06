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

#include "dump_klv.h"

#include <iostream>
#include <fstream>

#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block_formatter.h>
#include <vital/exceptions.h>
#include <vital/util/get_paths.h>
#include <vital/util/wrap_text_block.h>

#include <vital/algo/video_input.h>

#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>
#include <vital/plugin_loader/plugin_manager.h>

namespace kwiver {
namespace arrows {
namespace core {

namespace {

//+ maybe not needed
// ------------------------------------------------------------------
static kwiver::vital::config_block_sptr default_config()
{
  kwiver::vital::config_block_sptr config =
    kwiver::vital::config_block::empty_config( "dump_klv_tool" );

  config->set_value( "video_reader:type", "ffmpeg",
                     "Implementation for video reader." );
  config->set_value( "video_reader:vidl_ffmpeg:time_source",  "misp",
                     "Time source for reader." );

  kwiver::vital::algo::video_input::get_nested_algo_configuration(
    "video_reader", config, kwiver::vital::algo::video_input_sptr() );

  return config;
}

} // end namespace


// ----------------------------------------------------------------------------
void
dump_klv::
add_command_options()
{
  m_cmd_options->custom_help( wrap_text(
     "[options]  video-file\n"
     "This program displays the KLV metadata packets that are embedded "
     "in a video file."
                                ) );
  m_cmd_options->positional_help( "\n  video-file  - name of video file." );

  m_cmd_options->add_options()
    ( "h,help", "Display applet usage" )
    ( "c,config", "Configuration file for tool", cxxopts::value<std::string>() )
    ( "o,output", "Dump configuration to file and exit", cxxopts::value<std::string>() )
    ( "d,detail", "Display a detailed description of the metadata" )

    // positional parameters
    ( "video-file", "Video input file", cxxopts::value<std::string>())
    ;

  m_cmd_options->parse_positional("video-file");
}


// ============================================================================
dump_klv::
dump_klv()
{ }

// ----------------------------------------------------------------
/** Main entry. */
int
dump_klv::
run()
{
  const std::string opt_app_name = applet_name();
  std::string video_file;
  kwiver::vital::metadata_traits md_traits;

  auto& cmd_args = command_args();

  if ( cmd_args["help"].as<bool>() )
  {
    std::cout << m_cmd_options->help();
    return EXIT_SUCCESS;
  }

  if ( cmd_args.count("video-file") )
  {
    video_file = cmd_args["video-file"].as<std::string>();
  }
  else
  {
    std::cout << "Missing video file name.\n"
              << m_cmd_options->help();

    return EXIT_FAILURE;
  }

  // register the algorithm implementations
  const std::string rel_plugin_path = kwiver::vital::get_executable_path() + "/../lib/modules";
  kwiver::vital::plugin_manager::instance().add_search_path(rel_plugin_path);
  kwiver::vital::plugin_manager::instance().load_all_plugins();
  kwiver::vital::algo::video_input_sptr video_reader;
  kwiver::vital::config_block_sptr config = default_config();

  // If --config given, read in config file, merge in with default just generated
  if( cmd_args.count("config") )
  {
    config->merge_config( kwiver::vital::read_config_file( cmd_args["config"].as<std::string>() ) );
  }

  kwiver::vital::algo::video_input::set_nested_algo_configuration( "video_reader", config, video_reader );
  kwiver::vital::algo::video_input::get_nested_algo_configuration( "video_reader", config, video_reader );
  // Check to see if we are to dump config
  if ( cmd_args.count("output") )
  {
    const std::string out_file = cmd_args["output"].as<std::string>();
    std::ofstream fout( out_file.c_str() );
    if( ! fout )
    {
      std::cout << "Couldn't open \"" << out_file << "\" for writing.\n";
      return EXIT_FAILURE;
    }

    kwiver::vital::config_block_formatter fmt( config );
    fmt.print( fout );
    std::cout << "Wrote config to \"" << out_file << "\". Exiting.\n";
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
    std::cerr << "Video Exception-Couldn't open \"" << video_file << "\"" << std::endl
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  catch ( kwiver::vital::file_not_found_exception const& e )
  {
    std::cerr << "Couldn't open \"" << video_file << "\"" << std::endl
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
  kwiver::vital::wrap_text_block wtb;
  wtb.set_indent_string( "    " );

  while ( video_reader->next_frame( ts ) )
  {
    std::cout << "========== Read frame " << ts.get_frame()
              << " (index " << count << ") ==========" << std::endl;

    kwiver::vital::metadata_vector metadata = video_reader->frame_metadata();
    for( auto meta : metadata )
    {
      std::cout << "\n\n---------------- Metadata from: " << meta->timestamp() << std::endl;

      if ( cmd_args["detail"].as<bool>() )
      {
        for (const auto ix : *meta)
        {
          // process metada items
          const std::string name = ix.second->name();
          const kwiver::vital::any data = ix.second->data();
          const auto tag = ix.second->tag();
          const auto descrip = md_traits.tag_to_description( tag );

          std::cout
              << "Metadata item: " << name << std::endl
              << wtb.wrap_text( descrip )
              << "Data: <" << ix.second->type().name() << ">: "
              << kwiver::vital::metadata::format_string(ix.second->as_string())
              << std::endl;
        } // end for
      }
      else
      {
        print_metadata( std::cout, *meta );
      }

      ++count;
    } // end for over metadata collection vector

  } // end while over video

  std::cout << "-- End of video --\n";

  return EXIT_SUCCESS;
}

} } } // end namespace
