/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "convert.h"

#include <plugins/core/convert_annotations.h>
#include <plugins/core/python_script_applet.h>
#include <plugins/core/utilities_file.h>

#include <vital/algo/algorithm.txx>
#include <vital/algo/detected_object_set_input.h>
#include <vital/algo/detected_object_set_output.h>
#include <vital/algo/read_object_track_set.h>
#include <vital/algo/write_object_track_set.h>
#include <vital/plugin_management/plugin_manager.h>
#include <vital/logger/logger.h>

#include <kwiversys/SystemTools.hxx>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

namespace viame {
namespace tools {

namespace kv = kwiver::vital;
using ST = kwiversys::SystemTools;

namespace {

// -----------------------------------------------------------------------------
std::vector< std::string >
sorted_impl_names( std::vector< std::string > names )
{
  std::sort( names.begin(), names.end() );
  return names;
}

// -----------------------------------------------------------------------------
std::string
join( std::vector< std::string > const& items )
{
  std::ostringstream out;
  for( std::size_t i = 0; i < items.size(); ++i )
  {
    out << ( i ? ", " : "" ) << items[i];
  }
  return out.str();
}

// -----------------------------------------------------------------------------
// "reader:key=value" / "writer:key=value" / "key=value" (applies to both)
bool
parse_setting( std::string const& text,
               annotation_conversion_options& options )
{
  const std::size_t equals = text.find( '=' );
  if( equals == std::string::npos || equals == 0 )
  {
    return false;
  }
  std::string key = text.substr( 0, equals );
  const std::string value = text.substr( equals + 1 );

  if( key.rfind( "reader:", 0 ) == 0 )
  {
    options.reader_settings.emplace_back( key.substr( 7 ), value );
  }
  else if( key.rfind( "writer:", 0 ) == 0 )
  {
    options.writer_settings.emplace_back( key.substr( 7 ), value );
  }
  else
  {
    options.reader_settings.emplace_back( key, value );
    options.writer_settings.emplace_back( key, value );
  }
  return true;
}

// -----------------------------------------------------------------------------
// Preference between annotation files sharing a name: formats carrying the
// most detail win
int
format_priority( std::string const& format )
{
  static const std::vector< std::string > order =
    { "kw18", "cvat", "yolo", "habcam", "oceaneyes", "fishnet", "coco", "dive", "viame_csv" };
  auto itr = std::find( order.begin(), order.end(), format );
  return itr == order.end() ? 0 : static_cast< int >( itr - order.begin() ) + 1;
}

// -----------------------------------------------------------------------------
// Whether the raw command line carries options this applet does not define,
// which belong to the calibration converter
bool
has_foreign_options( std::vector< std::string > const& applet_args )
{
  static const std::set< std::string > known =
    { "--help", "--input", "--output", "--input-format", "--output-format",
      "--images", "--no-images", "--frate", "--frame-rate", "--frame-offset",
      "--setting", "--list-formats", "--calibration-help",
      "-h", "-i", "-o", "-s" };

  static const std::set< std::string > calibration =
    { "--camera-mode", "--image-width", "--image-height", "--left-cal",
      "--right-cal", "--pts", "--extrinsics-mode", "--left", "--right" };
  static const std::set< std::string > flags =
    { "--help", "-h", "--no-images", "--list-formats", "--calibration-help" };
  for( std::size_t i = 1; i < applet_args.size(); ++i )
  {
    const auto& arg = applet_args[i];
    if( arg == "--" ) { break; }
    if( arg.size() < 2 || arg[0] != '-' ) { continue; }
    const auto equals = arg.find( '=' );
    const auto name = arg.substr( 0, equals );
    if( calibration.count( name ) ) { return true; }
    if( !known.count( name ) )
    {
      // Short options may carry their value without a space.
      if( arg.size() > 2 && arg[1] != '-' &&
          (arg[1] == 'i' || arg[1] == 'o' || arg[1] == 's') ) { continue; }
      throw std::runtime_error( "Unknown conversion option: " + name );
    }
    if( !flags.count( name ) && equals == std::string::npos )
    {
      ++i; // A value such as -1 is not another option.
    }
  }
  return false;
}

// -----------------------------------------------------------------------------
// Hand everything to the calibration converter script, untouched
int
run_calibration_converter( std::vector< std::string > const& applet_args )
{
  const std::string script = find_tool_script( "convert.py" );
  if( script.empty() )
  {
    std::cerr << "Unable to locate convert.py; set VIAME_INSTALL or run from "
              << "an installed VIAME tree" << std::endl;
    return EXIT_FAILURE;
  }
  return run_tool_script( script,
    std::vector< std::string >( applet_args.begin() + 1, applet_args.end() ) );
}

// -----------------------------------------------------------------------------
std::string
describe( annotation_conversion_summary const& summary )
{
  std::ostringstream out;
  out << summary.reader << " -> " << summary.writer << ", "
      << summary.frames << " frame" << ( summary.frames == 1 ? "" : "s" ) << ", ";
  if( summary.used_tracks || summary.tracks > 0 )
  {
    out << summary.tracks << " track" << ( summary.tracks == 1 ? "" : "s" ) << ", ";
  }
  out << summary.detections << " detection" << ( summary.detections == 1 ? "" : "s" );
  if( !summary.frame_source.empty() )
  {
    out << ", frames from " << ST::GetFilenameName( summary.frame_source );
  }
  return out.str();
}

} // anonymous namespace

// =============================================================================
convert_applet
::convert_applet()
{
}

// -----------------------------------------------------------------------------
void
convert_applet
::add_command_options()
{
  m_cmd_options->custom_help( wrap_text(
    "[options] input output\n\n"
    "Convert annotation files between formats, singly or by folder, or "
    "convert camera calibration and registration files.\n\n"
    "Annotation inputs are recognised from their extension and content. When "
    "the input is a folder, every annotation file below it is converted into "
    "the output folder under the same relative path. Imagery found next to an "
    "annotation file (images in the same folder, or a video) supplies the frame "
    "names, count and timing of the output; without it, the frames come from "
    "the annotation file alone.\n\n"
    "Inputs that are not annotation files are passed to the calibration "
    "converter (stereo calibrations: npz, json, opencv, yml, mat, zed, cal; ITK "
    "transforms: h5); see --calibration-help for its options." ) );

  m_cmd_options->positional_help( "input output" );

  m_cmd_options->add_options()
    ( "h,help", "Display usage information",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "input", "Annotation file or folder to convert",
      ::cxxopts::value< std::string >()->default_value( "" ), "path" )
    ( "output", "Output file, or output folder when the input is a folder",
      ::cxxopts::value< std::string >()->default_value( "" ), "path" )
    ( "i,input-format", "Annotation reader to use (default: detect from the file)",
      ::cxxopts::value< std::string >()->default_value( "" ), "format" )
    ( "o,output-format", "Annotation writer to use (default: from the output "
      "extension: .csv viame_csv, .json coco, .dive.json dive, .kw18 kw18)",
      ::cxxopts::value< std::string >()->default_value( "" ), "format" )
    ( "images", "Image folder, image list or video the annotations belong to "
      "(default: whatever sits next to each annotation file)",
      ::cxxopts::value< std::string >()->default_value( "" ), "path" )
    ( "no-images", "Ignore imagery next to the annotations; frames come from "
      "the annotation file alone",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "frame-rate", "Frames per second for timestamps and, for videos, "
      "the rate the frames are numbered at (default: native)",
      ::cxxopts::value< std::string >()->default_value( "" ), "fps" )
    ( "frate", "Same as --frame-rate",
      ::cxxopts::value< std::string >()->default_value( "" ), "fps" )
    ( "frame-offset", "Value added to output frame numbers",
      ::cxxopts::value< int >()->default_value( "0" ), "count" )
    ( "s,setting", "Reader or writer configuration as key=value, optionally "
      "prefixed by reader: or writer:",
      ::cxxopts::value< std::vector< std::string > >(), "key=value" )
    ( "list-formats", "List the registered annotation readers and writers",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "calibration-help", "Show the calibration converter's options",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ;

  m_cmd_options->parse_positional( { "input", "output" } );
  m_cmd_options->allow_unrecognised_options();
}

// -----------------------------------------------------------------------------
int
convert_applet
::run()
{
  auto& cmd_args = command_args();
  kv::logger_handle_t logger = kv::get_logger( "viame.tools.convert" );

  const std::string input = cmd_args[ "input" ].as< std::string >();
  const std::string output = cmd_args[ "output" ].as< std::string >();
  const std::string input_format = cmd_args[ "input-format" ].as< std::string >();
  const std::string output_format = cmd_args[ "output-format" ].as< std::string >();
  if( cmd_args[ "calibration-help" ].as< bool >() )
  {
    return run_calibration_converter( { applet_name(), "--help" } );
  }

  if( cmd_args[ "help" ].as< bool >() )
  {
    std::cout << m_cmd_options->help() << std::endl;
    return EXIT_SUCCESS;
  }

  // Options this applet does not know belong to the calibration converter
  if( has_foreign_options( applet_args() ) )
  {
    return run_calibration_converter( applet_args() );
  }

  kv::plugin_manager::instance().load_all_plugins();

  if( cmd_args[ "list-formats" ].as< bool >() )
  {
    kv::plugin_manager& vpm = kv::plugin_manager::instance();
    std::cout << "Detection readers: " << join( sorted_impl_names(
      vpm.impl_names< kv::algo::detected_object_set_input >() ) ) << std::endl;
    std::cout << "Track readers:     " << join( sorted_impl_names(
      vpm.impl_names< kv::algo::read_object_track_set >() ) ) << std::endl;
    std::cout << "Detection writers: " << join( sorted_impl_names(
      vpm.impl_names< kv::algo::detected_object_set_output >() ) ) << std::endl;
    std::cout << "Track writers:     " << join( sorted_impl_names(
      vpm.impl_names< kv::algo::write_object_track_set >() ) ) << std::endl;
    return EXIT_SUCCESS;
  }

  if( input.empty() )
  {
    std::cerr << m_cmd_options->help() << std::endl;
    return EXIT_FAILURE;
  }

  const bool input_is_folder = does_folder_exist( input );
  const bool explicit_annotation_format =
    !input_format.empty() &&
    ( kv::has_algorithm_impl_name< kv::algo::read_object_track_set >( input_format ) ||
      kv::has_algorithm_impl_name< kv::algo::detected_object_set_input >( input_format ) );

  // A calibration can be an OpenCV directory containing intrinsics.yml and
  // extrinsics.yml.  Keep auto-detected annotation folders here, but hand a
  // directory with no annotation files to convert.py so it can recognize
  // calibration directories as it did before this became a C++ applet.
  if( input_is_folder && !explicit_annotation_format &&
      list_annotation_files( input, std::string() ).empty() )
  {
    return run_calibration_converter( applet_args() );
  }

  // Annotation files are converted here, anything else by the script
  if( !input_is_folder && !explicit_annotation_format )
  {
    if( !does_file_exist( input ) )
    {
      std::cerr << "Input " << input << " does not exist" << std::endl;
      return EXIT_FAILURE;
    }
    if( detect_annotation_format( input ).empty() )
    {
      return run_calibration_converter( applet_args() );
    }
  }

  annotation_conversion_options options;
  options.input_format = input_format.empty() ? "auto" : input_format;
  options.frame_offset = cmd_args[ "frame-offset" ].as< int >();

  std::string frate = cmd_args[ "frame-rate" ].as< std::string >();
  if( frate.empty() )
  {
    frate = cmd_args[ "frate" ].as< std::string >();
  }
  if( !frate.empty() )
  {
    try
    {
      options.frame_rate = std::stod( frate );
    }
    catch( std::exception const& )
    {
      std::cerr << "Invalid frame rate: " << frate << std::endl;
      return EXIT_FAILURE;
    }
  }

  if( cmd_args.count( "setting" ) )
  {
    for( auto const& setting : cmd_args[ "setting" ].as< std::vector< std::string > >() )
    {
      if( !parse_setting( setting, options ) )
      {
        std::cerr << "Settings must be key=value, got: " << setting << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  const bool no_images = cmd_args[ "no-images" ].as< bool >();
  const std::string images = cmd_args[ "images" ].as< std::string >();

  if( !images.empty() && !does_folder_exist( images ) && !does_file_exist( images ) )
  {
    std::cerr << "Image source " << images << " does not exist" << std::endl;
    return EXIT_FAILURE;
  }

  // Pairs of input and output files to convert
  std::vector< std::pair< std::string, std::string > > jobs;
  std::string writer_format = output_format;

  if( input_is_folder )
  {
    if( output.empty() )
    {
      std::cerr << "An output folder is required" << std::endl;
      return EXIT_FAILURE;
    }
    if( writer_format.empty() )
    {
      std::cerr << "An output format (-o) is required when converting a folder"
                << std::endl;
      return EXIT_FAILURE;
    }
    const std::string ext = extension_for_format( writer_format );
    if( ext.empty() )
    {
      std::cerr << "Unknown output format: " << writer_format << std::endl;
      return EXIT_FAILURE;
    }

    const std::string input_root = ST::CollapseFullPath( input );
    const std::string output_root = ST::CollapseFullPath( output );
    const auto files = list_annotation_files(
      input, explicit_annotation_format ? input_format : std::string() );

    // Several annotation files with one stem (groundtruth.csv next to
    // groundtruth.kw18) would land on the same output; keep the richest
    std::map< std::string, std::pair< std::string, int > > chosen;
    std::vector< std::string > targets;

    for( auto const& file : files )
    {
      const std::string full = ST::CollapseFullPath( file );
      std::string relative = full.substr( input_root.size() );
      while( !relative.empty() && ( relative[0] == '/' || relative[0] == '\\' ) )
      {
        relative = relative.substr( 1 );
      }
      const std::string target = append_path( output_root, replace_ext_with( relative, ext ) );
      if( ST::CollapseFullPath( target ) == full )
      {
        LOG_WARN( logger, "Skipping " << file << ": it would overwrite itself" );
        continue;
      }

      const int rank = format_priority( explicit_annotation_format ?
        input_format : detect_annotation_format( file ) );
      auto existing = chosen.find( target );
      if( existing == chosen.end() )
      {
        chosen[ target ] = { file, rank };
        targets.push_back( target );
      }
      else if( rank > existing->second.second )
      {
        std::cout << "Skipping " << existing->second.first << " in favour of "
                  << file << std::endl;
        existing->second = { file, rank };
      }
      else
      {
        std::cout << "Skipping " << file << " in favour of "
                  << existing->second.first << std::endl;
      }
    }

    for( auto const& target : targets )
    {
      jobs.emplace_back( chosen[ target ].first, target );
    }

    if( jobs.empty() )
    {
      std::cerr << "No annotation files found under " << input << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Converting " << jobs.size() << " annotation file"
              << ( jobs.size() == 1 ? "" : "s" ) << " from " << input
              << " into " << output << std::endl;
  }
  else
  {
    if( output.empty() )
    {
      std::cerr << "An output file is required" << std::endl;
      return EXIT_FAILURE;
    }
    if( writer_format.empty() )
    {
      writer_format = format_from_extension( output );
    }
    if( writer_format.empty() )
    {
      std::cerr << "Cannot tell the output format from " << output
                << "; give it with -o" << std::endl;
      return EXIT_FAILURE;
    }
    if( does_folder_exist( output ) )
    {
      const std::string ext = extension_for_format( writer_format );
      jobs.emplace_back( input, append_path( output,
        replace_ext_with( ST::GetFilenameName( input ), ext ) ) );
    }
    else
    {
      jobs.emplace_back( input, output );
    }
  }

  options.output_format = writer_format;

  std::size_t failures = 0;

  for( auto const& job : jobs )
  {
    annotation_conversion_options file_options = options;
    if( no_images )
    {
      file_options.frame_source.clear();
    }
    else if( !images.empty() )
    {
      file_options.frame_source = images;
    }
    else
    {
      file_options.frame_source = find_frame_source_alongside( job.first );
    }

    annotation_conversion_summary summary;
    if( convert_annotation_file( job.first, job.second, file_options, summary, logger ) )
    {
      std::cout << job.first << " -> " << job.second << " (" << describe( summary )
                << ")" << std::endl;
    }
    else
    {
      std::cerr << "Failed to convert " << job.first << std::endl;
      ++failures;
    }
  }

  if( failures )
  {
    std::cerr << failures << " of " << jobs.size() << " conversion"
              << ( jobs.size() == 1 ? "" : "s" ) << " failed" << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

} // namespace tools
} // namespace viame
