/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of pipeline-free annotation conversion
 */

#include "convert_annotations.h"
#include "utilities_file.h"

#include <vital/algo/algorithm.txx>
#include <vital/algo/detected_object_set_input.h>
#include <vital/algo/detected_object_set_output.h>
#include <vital/algo/read_object_track_set.h>
#include <vital/algo/write_object_track_set.h>
#include <vital/algo/video_input.h>
#include <vital/config/config_block.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/object_track_set.h>
#include <vital/types/timestamp.h>

#include <kwiversys/SystemTools.hxx>

#include <vital/internal/cereal/external/rapidjson/document.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <set>

namespace viame {

namespace kv = kwiver::vital;
using ST = kwiversys::SystemTools;

namespace {

// -----------------------------------------------------------------------------
std::string
lower_extension( std::string const& path )
{
  return to_lower( ST::GetFilenameLastExtension( path ) );
}

// -----------------------------------------------------------------------------
bool
has_extension( std::string const& path, std::vector< std::string > const& exts )
{
  const std::string ext = lower_extension( path );
  return std::find( exts.begin(), exts.end(), ext ) != exts.end();
}

// -----------------------------------------------------------------------------
std::string
read_head( std::string const& path, std::size_t max_bytes = 4096 )
{
  std::ifstream fin( path, std::ios::binary );
  if( !fin )
  {
    return {};
  }
  std::string head( max_bytes, '\0' );
  fin.read( &head[0], static_cast< std::streamsize >( max_bytes ) );
  head.resize( static_cast< std::size_t >( fin.gcount() ) );
  return head;
}

// -----------------------------------------------------------------------------
std::string
json_annotation_format( std::string const& content )
{
  // COCO documents carry annotations and categories; DIVE documents never
  // do, while COCO files from kwcoco may also carry a "tracks" table
  if( content.find( "\"annotations\"" ) != std::string::npos ||
      content.find( "\"categories\"" ) != std::string::npos )
  {
    return "coco";
  }
  if( content.find( "\"tracks\"" ) != std::string::npos ||
      content.find( "\"confidencePairs\"" ) != std::string::npos ||
      content.find( "\"features\"" ) != std::string::npos )
  {
    return "dive";
  }
  if( content.find( "\"images\"" ) != std::string::npos )
  {
    return "coco";
  }
  return {};
}

// -----------------------------------------------------------------------------
// Image file names from a COCO document, keyed by frame. The Python COCO
// reader cannot hand names back through the C++ interface, so they are read
// straight from the images table.
std::map< std::size_t, std::string >
coco_frame_names( std::string const& path )
{
  std::map< std::size_t, std::string > names;

  std::ifstream fin( path, std::ios::binary );
  if( !fin )
  {
    return names;
  }
  std::stringstream buffer;
  buffer << fin.rdbuf();

  rapidjson::Document doc;
  doc.Parse( buffer.str().c_str() );
  if( doc.HasParseError() || !doc.IsObject() ||
      !doc.HasMember( "images" ) || !doc[ "images" ].IsArray() )
  {
    return names;
  }

  std::size_t position = 0;
  for( auto const& image : doc[ "images" ].GetArray() )
  {
    if( !image.IsObject() )
    {
      continue;
    }
    std::size_t frame = position;
    if( image.HasMember( "frame_index" ) && image[ "frame_index" ].IsNumber() )
    {
      frame = static_cast< std::size_t >( std::max( 0, image[ "frame_index" ].GetInt() ) );
    }
    if( image.HasMember( "file_name" ) && image[ "file_name" ].IsString() )
    {
      names[ frame ] = image[ "file_name" ].GetString();
    }
    ++position;
  }
  return names;
}

// -----------------------------------------------------------------------------
template < typename ALGO >
std::shared_ptr< ALGO >
make_algorithm( std::string const& name,
                std::vector< std::pair< std::string, std::string > > const& settings,
                std::string const& role, kv::logger_handle_t logger )
{
  if( !kv::has_algorithm_impl_name< ALGO >( name ) )
  {
    return nullptr;
  }

  auto algo = kv::create_algorithm< ALGO >( name );
  if( !algo )
  {
    return nullptr;
  }

  kv::config_block_sptr config = algo->get_configuration();
  for( auto const& setting : settings )
  {
    config->set_value( setting.first, setting.second );
  }
  if( !algo->check_configuration( config ) )
  {
    LOG_ERROR( logger, "Invalid configuration for " << role << " '" << name << "'" );
    return nullptr;
  }
  algo->set_configuration( config );
  return algo;
}

// -----------------------------------------------------------------------------
// Set a writer option only when the implementation declares it
template < typename ALGO >
void
set_optional_setting( std::shared_ptr< ALGO > const& algo,
                      std::string const& key, std::string const& value )
{
  kv::config_block_sptr config = algo->get_configuration();
  if( config->has_value( key ) )
  {
    config->set_value( key, value );
    algo->set_configuration( config );
  }
}

// -----------------------------------------------------------------------------
// Frames the output should cover, with their identifiers and times
struct frame_plan
{
  bool valid = false;
  bool from_video = false;
  std::string stream_id;
  std::vector< std::string > names;
  std::vector< double > times;   // seconds, negative when unknown
};

// -----------------------------------------------------------------------------
void
apply_frame_rate( frame_plan& plan, double frame_rate )
{
  if( frame_rate <= 0.0 )
  {
    return;
  }
  plan.times.resize( plan.names.size() );
  for( std::size_t k = 0; k < plan.times.size(); ++k )
  {
    plan.times[k] = static_cast< double >( k ) / frame_rate;
  }
}

// -----------------------------------------------------------------------------
frame_plan
plan_from_images( std::vector< std::string > const& images, double frame_rate )
{
  frame_plan plan;
  plan.valid = !images.empty();
  for( auto const& image : images )
  {
    plan.names.push_back( ST::GetFilenameName( image ) );
  }
  plan.times.assign( plan.names.size(), -1.0 );
  apply_frame_rate( plan, frame_rate );
  return plan;
}

// -----------------------------------------------------------------------------
frame_plan
plan_from_list( std::string const& list_file, double frame_rate )
{
  std::vector< std::string > images;
  std::ifstream fin( list_file );
  std::string line;
  while( std::getline( fin, line ) )
  {
    std::string trimmed;
    if( trim_line( line, trimmed ) )
    {
      images.push_back( trimmed );
    }
  }
  return plan_from_images( images, frame_rate );
}

// -----------------------------------------------------------------------------
frame_plan
plan_from_video( std::string const& video, double frame_rate, kv::logger_handle_t logger )
{
  frame_plan plan;
  plan.from_video = true;
  plan.stream_id = ST::GetFilenameName( video );

  kv::algo::video_input_sptr reader;
  for( auto const& impl : { "ffmpeg", "vidl_ffmpeg", "ffmpeg_clip" } )
  {
    if( kv::has_algorithm_impl_name< kv::algo::video_input >( impl ) )
    {
      reader = kv::create_algorithm< kv::algo::video_input >( impl );
      if( reader )
      {
        break;
      }
    }
  }

  if( !reader )
  {
    LOG_WARN( logger, "No video reader available; frames for " << video
                      << " are numbered from the annotations alone" );
    return plan;
  }

  std::size_t count = 0;
  double native_rate = 0.0;
  try
  {
    reader->open( video );
    count = reader->num_frames();
    native_rate = reader->frame_rate();

    if( native_rate <= 0.0 && count > 1 && reader->next_frame() )
    {
      const double t0 = reader->frame_timestamp().get_time_seconds();
      if( reader->next_frame() )
      {
        const double t1 = reader->frame_timestamp().get_time_seconds();
        if( t1 > t0 )
        {
          native_rate = 1.0 / ( t1 - t0 );
        }
      }
    }
    reader->close();
  }
  catch( std::exception const& e )
  {
    LOG_WARN( logger, "Unable to read " << video << ": " << e.what()
                      << "; frames are numbered from the annotations alone" );
    return plan;
  }

  if( count == 0 )
  {
    LOG_WARN( logger, "Video " << video << " reports no frames" );
    return plan;
  }

  double output_rate = native_rate;
  std::size_t output_count = count;
  if( frame_rate > 0.0 && native_rate > 0.0 && frame_rate < native_rate )
  {
    output_rate = frame_rate;
    output_count = static_cast< std::size_t >(
      std::floor( static_cast< double >( count ) * frame_rate / native_rate ) );
  }
  else if( frame_rate > 0.0 )
  {
    output_rate = frame_rate;
  }

  plan.valid = true;
  plan.names.assign( output_count, plan.stream_id );
  plan.times.assign( output_count, -1.0 );
  apply_frame_rate( plan, output_rate );

  LOG_INFO( logger, "Video " << plan.stream_id << ": " << count << " frames at "
                    << native_rate << " fps, writing " << output_count
                    << " frames at " << output_rate << " fps" );
  return plan;
}

// -----------------------------------------------------------------------------
frame_plan
plan_frames( annotation_conversion_options const& options, kv::logger_handle_t logger )
{
  const std::string& source = options.frame_source;
  if( source.empty() )
  {
    return {};
  }

  if( does_folder_exist( source ) )
  {
    std::vector< std::string > images;
    list_files_in_folder( source, images, false, default_image_extensions() );
    std::sort( images.begin(), images.end() );
    frame_plan plan = plan_from_images( images, options.frame_rate );
    if( !plan.valid )
    {
      LOG_WARN( logger, "No images found in " << source );
    }
    return plan;
  }

  if( !does_file_exist( source ) )
  {
    LOG_WARN( logger, "Frame source " << source << " does not exist" );
    return {};
  }

  if( has_extension( source, default_video_extensions() ) )
  {
    return plan_from_video( source, options.frame_rate, logger );
  }

  frame_plan plan = plan_from_list( source, options.frame_rate );
  if( !plan.valid )
  {
    LOG_WARN( logger, "Image list " << source << " is empty" );
  }
  return plan;
}

// -----------------------------------------------------------------------------
kv::timestamp
make_timestamp( std::size_t frame, frame_plan const& plan )
{
  kv::timestamp ts;
  ts.set_frame( static_cast< kv::frame_id_t >( frame ) );
  if( frame < plan.times.size() && plan.times[frame] >= 0.0 )
  {
    ts.set_time_seconds( plan.times[frame] );
  }
  return ts;
}

// -----------------------------------------------------------------------------
std::string
frame_name( std::size_t frame, frame_plan const& plan,
            std::map< std::size_t, std::string > const& names_from_file )
{
  if( plan.valid && frame < plan.names.size() )
  {
    return plan.names[frame];
  }
  auto itr = names_from_file.find( frame );
  return itr == names_from_file.end() ? std::string() : itr->second;
}

// -----------------------------------------------------------------------------
// Frame identifiers recorded in the annotation file itself, keyed by frame
std::map< std::size_t, std::string >
harvest_frame_names( std::string const& input_path, std::string const& format,
                     annotation_conversion_options const& options,
                     kv::logger_handle_t logger )
{
  std::map< std::size_t, std::string > names;

  if( format == "coco" )
  {
    return coco_frame_names( input_path );
  }

  auto reader = make_algorithm< kv::algo::detected_object_set_input >(
    format, options.reader_settings, "reader", logger );
  if( !reader )
  {
    return names;
  }

  try
  {
    reader->open( input_path );
    std::string name;
    std::size_t frame = 0;
    for(;;)
    {
      auto set = std::make_shared< kv::detected_object_set >();
      if( !reader->read_set( set, name ) )
      {
        break;
      }
      if( !name.empty() )
      {
        names[frame] = name;
      }
      name.clear();
      ++frame;
    }
    reader->close();
  }
  catch( std::exception const& e )
  {
    LOG_DEBUG( logger, "Frame names unavailable from " << input_path << ": " << e.what() );
  }
  return names;
}

// -----------------------------------------------------------------------------
std::vector< kv::track_sptr >
tracks_from_detections( std::map< std::size_t, kv::detected_object_set_sptr > const& frames )
{
  std::vector< kv::track_sptr > tracks;
  kv::track_id_t next_id = 1;
  for( auto const& item : frames )
  {
    if( !item.second )
    {
      continue;
    }
    for( auto const& det : *item.second )
    {
      auto track = kv::track::create();
      track->set_id( next_id++ );
      track->append( std::make_shared< kv::object_track_state >(
        static_cast< kv::frame_id_t >( item.first ),
        static_cast< kv::time_usec_t >( item.first ), det ) );
      tracks.push_back( track );
    }
  }
  return tracks;
}

} // anonymous namespace

// =============================================================================
std::vector< std::string > const&
default_image_extensions()
{
  static const std::vector< std::string > exts =
    { ".bmp", ".dds", ".gif", ".heic", ".jpg", ".jpeg", ".png", ".psd", ".psp",
      ".pspimage", ".tga", ".thm", ".tif", ".tiff", ".sgi", ".pgm", ".ppm" };
  return exts;
}

// -----------------------------------------------------------------------------
std::vector< std::string > const&
default_video_extensions()
{
  static const std::vector< std::string > exts =
    { ".3qp", ".3g2", ".amv", ".asf", ".avi", ".drc", ".f4v", ".f4p", ".flv",
      ".m4v", ".mkv", ".mp4", ".m4p", ".mpg", ".mpg2", ".mp2", ".mpeg", ".mpe",
      ".mpv", ".mng", ".mts", ".m2ts", ".mov", ".mxf", ".nsv", ".ogv", ".qt",
      ".rm", ".rmvb", ".svi", ".webm", ".wmv", ".vob", ".ts" };
  return exts;
}

// -----------------------------------------------------------------------------
std::string
format_from_extension( std::string const& path_or_ext )
{
  std::string ext = to_lower( path_or_ext );
  if( ext.find( '.' ) != 0 || ext.find( '/' ) != std::string::npos ||
      ext.find( '\\' ) != std::string::npos )
  {
    if( ends_with_ci( ext, ".dive.json" ) ) { return "dive"; }
    if( ends_with_ci( ext, ".coco.json" ) ) { return "coco"; }
    ext = lower_extension( path_or_ext );
  }
  if( ext == ".csv" ) { return "viame_csv"; }
  if( ext == ".json" ) { return "coco"; }
  if( ext == ".kw18" ) { return "kw18"; }
  if( ext == ".xml" ) { return "cvat"; }
  return {};
}

// -----------------------------------------------------------------------------
std::string
extension_for_format( std::string const& format )
{
  const std::string name = to_lower( format );
  if( name == "viame_csv" || name == "csv" || name == "habcam" ||
      name == "oceaneyes" || name == "fishnet" )
  {
    return ".csv";
  }
  if( name == "coco" || name == "dive" ) { return ".json"; }
  if( name == "kw18" ) { return ".kw18"; }
  if( name == "cvat" ) { return ".xml"; }
  if( name == "yolo" ) { return ".txt"; }
  return {};
}

// -----------------------------------------------------------------------------
std::string
detect_annotation_format( std::string const& path )
{
  if( !does_file_exist( path ) )
  {
    return {};
  }

  if( ends_with_ci( path, ".dive.json" ) ) { return "dive"; }
  if( ends_with_ci( path, ".coco.json" ) ) { return "coco"; }

  const std::string ext = lower_extension( path );
  if( ext == ".csv" ) { return "viame_csv"; }
  if( ext == ".kw18" ) { return "kw18"; }
  if( ext == ".json" ) { return json_annotation_format( read_head( path ) ); }
  if( ext == ".xml" )
  {
    return read_head( path ).find( "<annotations" ) != std::string::npos ? "cvat" : "";
  }
  return {};
}

// -----------------------------------------------------------------------------
std::vector< std::string >
list_annotation_files( std::string const& folder, std::string const& format )
{
  std::vector< std::string > files;
  std::vector< std::string > extensions;

  if( !format.empty() )
  {
    const std::string ext = extension_for_format( format );
    if( !ext.empty() )
    {
      extensions.push_back( ext );
    }
  }
  if( extensions.empty() )
  {
    extensions = { ".csv", ".json", ".kw18", ".xml" };
  }

  std::vector< std::string > candidates;
  list_files_in_folder( folder, candidates, true, extensions );
  std::sort( candidates.begin(), candidates.end() );

  for( auto const& candidate : candidates )
  {
    const std::string name = ST::GetFilenameName( candidate );
    if( name.empty() || name[0] == '.' )
    {
      continue;
    }
    if( format.empty() && detect_annotation_format( candidate ).empty() )
    {
      continue;
    }
    files.push_back( candidate );
  }
  return files;
}

// -----------------------------------------------------------------------------
std::string
find_frame_source_alongside( std::string const& annotation_path )
{
  const std::string full = ST::CollapseFullPath( annotation_path );
  const std::string folder = ST::GetFilenamePath( full );
  const std::string base = to_lower( ST::GetFilenameWithoutLastExtension( full ) );

  std::vector< std::string > images, videos;
  list_files_in_folder( folder, images, false, default_image_extensions() );
  list_files_in_folder( folder, videos, false, default_video_extensions() );
  std::sort( videos.begin(), videos.end() );

  if( !videos.empty() )
  {
    for( auto const& video : videos )
    {
      const std::string video_base =
        to_lower( ST::GetFilenameWithoutLastExtension( video ) );
      if( video_base == base || base.rfind( video_base, 0 ) == 0 )
      {
        return video;
      }
    }
    if( videos.size() == 1 && images.empty() )
    {
      return videos.front();
    }
  }

  if( !images.empty() )
  {
    return folder;
  }
  return {};
}

// =============================================================================
bool
convert_annotation_file( std::string const& input_path,
                         std::string const& output_path,
                         annotation_conversion_options const& options,
                         annotation_conversion_summary& summary,
                         kv::logger_handle_t logger )
{
  summary = annotation_conversion_summary();

  if( !does_file_exist( input_path ) )
  {
    LOG_ERROR( logger, "Input " << input_path << " does not exist" );
    return false;
  }

  std::string input_format = to_lower( options.input_format );
  if( input_format.empty() || input_format == "auto" )
  {
    input_format = detect_annotation_format( input_path );
    if( input_format.empty() )
    {
      input_format = "auto";
    }
  }

  const std::string output_format = to_lower( options.output_format );
  if( output_format.empty() )
  {
    LOG_ERROR( logger, "No output format given" );
    return false;
  }

  const bool track_reader_available =
    kv::has_algorithm_impl_name< kv::algo::read_object_track_set >( input_format );
  const bool detection_reader_available =
    kv::has_algorithm_impl_name< kv::algo::detected_object_set_input >( input_format );
  const bool track_writer_available =
    kv::has_algorithm_impl_name< kv::algo::write_object_track_set >( output_format );
  const bool detection_writer_available =
    kv::has_algorithm_impl_name< kv::algo::detected_object_set_output >( output_format );

  if( !track_reader_available && !detection_reader_available )
  {
    LOG_ERROR( logger, "No reader for annotation format '" << input_format << "'" );
    return false;
  }
  if( !track_writer_available && !detection_writer_available )
  {
    LOG_ERROR( logger, "No writer for annotation format '" << output_format << "'" );
    return false;
  }

  summary.reader = input_format;
  summary.writer = output_format;
  summary.frame_source = options.frame_source;

  const frame_plan plan = plan_frames( options, logger );

  // Frame identifiers recorded in the file, used when no data is given
  std::map< std::size_t, std::string > names_from_file;
  if( !plan.valid && detection_reader_available )
  {
    names_from_file = harvest_frame_names( input_path, input_format, options, logger );
  }

  const std::string output_folder = ST::GetFilenamePath( ST::CollapseFullPath( output_path ) );
  if( !output_folder.empty() && !does_folder_exist( output_folder ) )
  {
    create_folder( output_folder );
  }

  // ---- read ---------------------------------------------------------------
  std::map< kv::track_id_t, kv::track_sptr > tracks_by_id;
  std::map< std::size_t, kv::detected_object_set_sptr > detections_by_frame;
  std::size_t last_frame = 0;
  bool have_frames = false;

  try
  {
    if( track_reader_available )
    {
      auto reader = make_algorithm< kv::algo::read_object_track_set >(
        input_format, options.reader_settings, "reader", logger );
      if( !reader )
      {
        return false;
      }
      // Whole-file reads where the reader offers them; streaming readers
      // hand out one frame per call and some never signal the end
      set_optional_setting< kv::algo::read_object_track_set >( reader, "batch_load", "true" );

      reader->open( input_path );
      std::size_t idle_reads = 0;
      static const std::size_t max_idle_reads = 100000;
      for(;;)
      {
        // A live set: Python readers fill it in place, C++ ones replace it
        auto set = std::make_shared< kv::object_track_set >();
        if( !reader->read_set( set ) )
        {
          break;
        }
        if( !set || set->tracks().empty() )
        {
          if( ++idle_reads > max_idle_reads )
          {
            LOG_WARN( logger, "Reader '" << input_format << "' returned "
                              << max_idle_reads << " empty frames in a row; stopping" );
            break;
          }
          continue;
        }
        idle_reads = 0;
        for( auto const& track : set->tracks() )
        {
          if( track && !track->empty() )
          {
            tracks_by_id[ track->id() ] = track;
          }
        }
      }
      reader->close();
      summary.used_tracks = true;

      for( auto const& item : tracks_by_id )
      {
        for( auto const& state : *item.second )
        {
          const std::size_t frame = static_cast< std::size_t >(
            std::max< kv::frame_id_t >( 0, state->frame() ) );
          last_frame = have_frames ? std::max( last_frame, frame ) : frame;
          have_frames = true;
          ++summary.detections;
        }
      }
      summary.tracks = tracks_by_id.size();
    }
    else
    {
      auto reader = make_algorithm< kv::algo::detected_object_set_input >(
        input_format, options.reader_settings, "reader", logger );
      if( !reader )
      {
        return false;
      }
      reader->open( input_path );

      if( plan.valid )
      {
        // Look each frame up by its identifier, as a pipeline would
        for( std::size_t frame = 0; frame < plan.names.size(); ++frame )
        {
          std::string name = plan.names[frame];
          auto set = std::make_shared< kv::detected_object_set >();
          if( !reader->read_set( set, name ) )
          {
            break;
          }
          if( set && set->size() > 0 )
          {
            detections_by_frame[frame] = set;
            summary.detections += set->size();
          }
        }
        last_frame = plan.names.empty() ? 0 : plan.names.size() - 1;
        have_frames = !plan.names.empty();
      }
      else
      {
        std::string name;
        std::size_t frame = 0;
        for(;;)
        {
          auto set = std::make_shared< kv::detected_object_set >();
          if( !reader->read_set( set, name ) )
          {
            break;
          }
          if( !name.empty() )
          {
            names_from_file[frame] = name;
          }
          if( set && set->size() > 0 )
          {
            detections_by_frame[frame] = set;
            summary.detections += set->size();
          }
          name.clear();
          last_frame = frame;
          have_frames = true;
          ++frame;
        }
      }
      reader->close();
    }
  }
  catch( std::exception const& e )
  {
    LOG_ERROR( logger, "Reading " << input_path << " failed: " << e.what() );
    return false;
  }

  // ---- frames to write ----------------------------------------------------
  std::size_t frame_count = 0;
  if( plan.valid )
  {
    frame_count = plan.names.size();
    if( have_frames && last_frame + 1 > frame_count )
    {
      LOG_WARN( logger, "Annotations reference frame " << last_frame
                        << " but the data only has " << frame_count
                        << " frames; extra frames are kept" );
      frame_count = last_frame + 1;
    }
  }
  else if( have_frames )
  {
    frame_count = last_frame + 1;
  }
  summary.frames = frame_count;

  const std::string offset = std::to_string( options.frame_offset );

  // ---- write --------------------------------------------------------------
  try
  {
    if( summary.used_tracks )
    {
      // Tracks active on each frame, for writers that stream per frame
      std::map< std::size_t, std::vector< kv::track_sptr > > tracks_by_frame;
      for( auto const& item : tracks_by_id )
      {
        for( auto const& state : *item.second )
        {
          tracks_by_frame[ static_cast< std::size_t >(
            std::max< kv::frame_id_t >( 0, state->frame() ) ) ].push_back( item.second );
        }
      }

      if( track_writer_available )
      {
        auto writer = make_algorithm< kv::algo::write_object_track_set >(
          output_format, options.writer_settings, "writer", logger );
        if( !writer )
        {
          return false;
        }
        if( options.frame_offset != 0 )
        {
          set_optional_setting< kv::algo::write_object_track_set >(
            writer, "frame_id_adjustment", offset );
        }
        if( plan.from_video && !plan.stream_id.empty() )
        {
          set_optional_setting< kv::algo::write_object_track_set >(
            writer, "stream_identifier", plan.stream_id );
        }
        writer->open( output_path );
        for( std::size_t frame = 0; frame < frame_count; ++frame )
        {
          auto itr = tracks_by_frame.find( frame );
          auto set = std::make_shared< kv::object_track_set >(
            itr == tracks_by_frame.end() ? std::vector< kv::track_sptr >() : itr->second );
          writer->write_set( set, make_timestamp( frame, plan ),
                             frame_name( frame, plan, names_from_file ) );
        }
        writer->close();
      }
      else
      {
        auto writer = make_algorithm< kv::algo::detected_object_set_output >(
          output_format, options.writer_settings, "writer", logger );
        if( !writer )
        {
          return false;
        }
        if( options.frame_offset != 0 )
        {
          set_optional_setting< kv::algo::detected_object_set_output >(
            writer, "frame_id_adjustment", offset );
        }
        writer->open( output_path );
        for( std::size_t frame = 0; frame < frame_count; ++frame )
        {
          auto set = std::make_shared< kv::detected_object_set >();
          auto itr = tracks_by_frame.find( frame );
          if( itr != tracks_by_frame.end() )
          {
            for( auto const& track : itr->second )
            {
              auto state_itr = track->find( static_cast< kv::frame_id_t >( frame ) );
              if( state_itr == track->end() )
              {
                continue;
              }
              auto const* state =
                dynamic_cast< kv::object_track_state const* >( state_itr->get() );
              if( state && state->detection() )
              {
                set->add( std::const_pointer_cast< kv::detected_object >( state->detection() ) );
              }
            }
          }
          writer->write_set( set, frame_name( frame, plan, names_from_file ) );
        }
        writer->close();
      }
    }
    else if( detection_writer_available )
    {
      auto writer = make_algorithm< kv::algo::detected_object_set_output >(
        output_format, options.writer_settings, "writer", logger );
      if( !writer )
      {
        return false;
      }
      if( options.frame_offset != 0 )
      {
        set_optional_setting< kv::algo::detected_object_set_output >(
          writer, "frame_id_adjustment", offset );
      }
      writer->open( output_path );
      for( std::size_t frame = 0; frame < frame_count; ++frame )
      {
        auto itr = detections_by_frame.find( frame );
        auto set = itr == detections_by_frame.end()
          ? std::make_shared< kv::detected_object_set >() : itr->second;
        writer->write_set( set, frame_name( frame, plan, names_from_file ) );
      }
      writer->close();
    }
    else
    {
      auto writer = make_algorithm< kv::algo::write_object_track_set >(
        output_format, options.writer_settings, "writer", logger );
      if( !writer )
      {
        return false;
      }
      if( options.frame_offset != 0 )
      {
        set_optional_setting< kv::algo::write_object_track_set >(
          writer, "frame_id_adjustment", offset );
      }
      if( plan.from_video && !plan.stream_id.empty() )
      {
        set_optional_setting< kv::algo::write_object_track_set >(
          writer, "stream_identifier", plan.stream_id );
      }
      const auto tracks = tracks_from_detections( detections_by_frame );
      summary.tracks = tracks.size();
      std::map< std::size_t, std::vector< kv::track_sptr > > tracks_by_frame;
      for( auto const& track : tracks )
      {
        tracks_by_frame[ static_cast< std::size_t >( track->first_frame() ) ].push_back( track );
      }
      writer->open( output_path );
      for( std::size_t frame = 0; frame < frame_count; ++frame )
      {
        auto itr = tracks_by_frame.find( frame );
        auto set = std::make_shared< kv::object_track_set >(
          itr == tracks_by_frame.end() ? std::vector< kv::track_sptr >() : itr->second );
        writer->write_set( set, make_timestamp( frame, plan ),
                           frame_name( frame, plan, names_from_file ) );
      }
      writer->close();
    }
  }
  catch( std::exception const& e )
  {
    LOG_ERROR( logger, "Writing " << output_path << " failed: " << e.what() );
    return false;
  }

  return true;
}

} // end namespace viame
