// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
///
/// This program reads a video and extracts all the KLV metadata.

#include "dump_klv.h"

#include <iostream>
#include <fstream>

#include <vital/algo/metadata_map_io.h>
#include <vital/algo/video_input.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <vital/config/config_block.h>
#include <vital/config/config_block_io.h>
#include <vital/config/config_block_formatter.h>

#include <vital/util/get_paths.h>
#include <vital/util/wrap_text_block.h>

#include <vital/types/metadata.h>
#include <vital/types/metadata_map.h>
#include <vital/types/metadata_traits.h>

#include <vital/exceptions.h>

namespace kv = kwiver::vital;
namespace kva = kwiver::vital::algo;

namespace kwiver {
namespace arrows {
namespace core {

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
    ( "c,config", "Configuration file for tool", cxxopts::value< std::string >() )
    ( "o,output", "Dump configuration to file and exit", cxxopts::value< std::string >() )
    ( "l,log", "Log metadata to a file. This requires the JSON serialization plugin. "
      "The file is structured as an array of frames where each frame contains an array "
      "of metadata packets associated with that frame. Each packet is an "
      "array of metadata fields. Alternatively, the configuration file, "
      "dump_klv.conf, can be updated to use CSV instead.",
      cxxopts::value< std::string >() )
    ( "d,detail", "Display a detailed description of the metadata" )
    ( "q,quiet", "Do not show metadata. Overrides -d/--detail." )

    // positional parameters
    ( "video-file", "Video input file", cxxopts::value< std::string >() )
    ;

  m_cmd_options->parse_positional("video-file");
}

// ----------------------------------------------------------------------------
dump_klv
::dump_klv()
{
}

// ----------------------------------------------------------------------------
int
dump_klv
::run()
{
  const std::string opt_app_name = applet_name();
  std::string video_file;
  kv::metadata_traits md_traits;

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

  kva::video_input_sptr video_reader;
  kva::metadata_map_io_sptr metadata_serializer_ptr;
  auto config = this->find_configuration("applets/dump_klv.conf");

  // If --config given, read in config file, merge in with default just generated
  if( cmd_args.count("config") )
  {
    config->merge_config( kv::read_config_file( cmd_args["config"].as<std::string>() ) );
  }

  kva::video_input::set_nested_algo_configuration(
    "video_reader", config, video_reader );
  kva::video_input::get_nested_algo_configuration(
    "video_reader", config, video_reader );
  kva::metadata_map_io::set_nested_algo_configuration(
    "metadata_serializer", config, metadata_serializer_ptr );
  kva::metadata_map_io::get_nested_algo_configuration(
    "metadata_serializer", config, metadata_serializer_ptr );

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

    kv::config_block_formatter fmt( config );
    fmt.print( fout );
    std::cout << "Wrote config to \"" << out_file << "\". Exiting.\n";
    return EXIT_SUCCESS;
  }

  if( !kva::video_input::check_nested_algo_configuration(
         "video_reader", config ) )
  {
    std::cerr << "Invalid video_reader config" << std::endl;
    return EXIT_FAILURE;
  }

  if ( !kva::metadata_map_io::check_nested_algo_configuration(
          "metadata_serializer", config ) )
  {
    std::cerr << "Invalid metadata_serializer config" << std::endl;
    return EXIT_FAILURE;
  }

  // instantiate a video reader
  try
  {
    video_reader->open( video_file );
  }
  catch ( kv::video_exception const& e )
  {
    std::cerr << "Video Exception-Couldn't open \"" << video_file << "\"" << std::endl
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  catch ( kv::file_not_found_exception const& e )
  {
    std::cerr << "Couldn't open \"" << video_file << "\"" << std::endl
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  auto const& caps = video_reader->get_implementation_capabilities();
  if ( !caps.capability( kva::video_input::HAS_METADATA ) )
  {
    std::cerr << "No metadata stream found in " << video_file << '\n';
    return EXIT_FAILURE;
  }

  int count(1);
  kv::image_container_sptr frame;
  kv::timestamp ts;
  kv::wrap_text_block wtb;
  kv::metadata_map::map_metadata_t frame_metadata;

  wtb.set_indent_string( "    " );

  // Avoid repeated dictionary access
  auto const detail = cmd_args[ "detail" ].as< bool >();
  auto const quiet = cmd_args[ "quiet" ].as< bool >();
  auto const log = ( cmd_args.count( "log" ) > 0 );

  while ( video_reader->next_frame( ts ) )
  {
    if ( !quiet )
    {
      std::cout << "========== Read frame " << ts.get_frame()
                << " (index " << count << ") ==========" << std::endl;
    }

    kv::metadata_vector metadata = video_reader->frame_metadata();

    if ( log )
    {
      // Add the (frame number, vector of metadata packets) item
      frame_metadata.insert( { ts.get_frame(), metadata } );
    }

    if ( !quiet )
    {
      for ( auto const& meta : metadata )
      {
        std::cout << "\n\n---------------- Metadata from: "
                  << meta->timestamp() << std::endl;

        if ( detail )
        {
          for ( auto const& ix : *meta )
          {
            // process metada items
            auto const& name = ix.second->name();
            auto const& data = ix.second->data();
            auto const& tag = ix.second->tag();
            auto const& descrip = md_traits.tag_to_description( tag );

            std::cout
                << "Metadata item: " << name << std::endl
                << wtb.wrap_text( descrip )
                << "Data: <" << ix.second->type().name() << ">: "
                << kv::metadata::format_string(ix.second->as_string())
                << std::endl;
          }
        }
        else
        {
          print_metadata( std::cout, *meta );
        }
      } // end for over metadata collection vector
    } // The end of not quiet
    ++count;
  } // end while over video

  if ( log )
  {
    const std::string out_file = cmd_args["log"].as<std::string>();
    std::ofstream fout( out_file.c_str() );
    if( ! fout )
    {
      std::cout << "Couldn't open \"" << out_file << "\" for writing.\n";
      return EXIT_FAILURE;
    }

    kv::metadata_map_sptr mms = std::make_shared<kv::simple_metadata_map>(
        kv::simple_metadata_map( frame_metadata ) );

    metadata_serializer_ptr->save( out_file, mms );

    std::cout << "Wrote KLV log to \"" << out_file << "\".\n";
  }

  std::cout << "-- End of video --\n";

  return EXIT_SUCCESS;
}

} // namespace core

} // namespace arrows

} // namespace kwiver
