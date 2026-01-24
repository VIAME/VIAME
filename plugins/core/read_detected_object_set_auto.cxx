/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_detected_object_set_auto
 */

#include "read_detected_object_set_auto.h"

#include "read_detected_object_set_cvat.h"
#include "read_detected_object_set_dive.h"
#include "read_detected_object_set_viame_csv.h"
#include "read_detected_object_set_yolo.h"
#include "utilities_file.h"

#include <vital/algo/detected_object_set_input.h>
#include <vital/exceptions.h>

#include <kwiversys/SystemTools.hxx>

#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>


namespace viame {

// -----------------------------------------------------------------------------------
// Helper functions for format detection
// -----------------------------------------------------------------------------------

namespace {

// Read first N bytes of a file for content inspection
std::string read_file_header( std::string const& filename, size_t max_bytes = 4096 )
{
  std::ifstream ifs( filename );
  if( !ifs )
  {
    return "";
  }

  std::string content;
  content.resize( max_bytes );
  ifs.read( &content[0], max_bytes );
  content.resize( ifs.gcount() );
  return content;
}

// Detect JSON format (DIVE vs COCO) by inspecting content
std::string detect_json_format( std::string const& content )
{
  // DIVE format has "tracks" as a top-level key with track objects
  // COCO format has "images", "annotations", and "categories" as top-level keys

  // Look for DIVE-specific patterns
  bool has_tracks = content.find( "\"tracks\"" ) != std::string::npos;
  bool has_features = content.find( "\"features\"" ) != std::string::npos;
  bool has_confidence_pairs = content.find( "\"confidencePairs\"" ) != std::string::npos;

  // Look for COCO-specific patterns
  bool has_images = content.find( "\"images\"" ) != std::string::npos;
  bool has_annotations = content.find( "\"annotations\"" ) != std::string::npos;
  bool has_categories = content.find( "\"categories\"" ) != std::string::npos;

  // DIVE format detection
  if( has_tracks && ( has_features || has_confidence_pairs ) )
  {
    return "dive";
  }

  // COCO format detection
  if( has_images && has_annotations && has_categories )
  {
    return "coco";
  }

  // If we have tracks but not COCO fields, assume DIVE
  if( has_tracks )
  {
    return "dive";
  }

  // If we have COCO-like fields, assume COCO
  if( has_annotations || ( has_images && has_categories ) )
  {
    return "coco";
  }

  // Default to COCO as it's more common
  return "coco";
}

} // anonymous namespace


// -----------------------------------------------------------------------------------
class read_detected_object_set_auto::priv
{
public:
  priv( read_detected_object_set_auto* parent )
    : m_parent( parent )
    , m_detected_format( "" )
  { }

  ~priv() { }

  std::string detect_format( std::string const& filename );

  read_detected_object_set_auto* m_parent;
  std::string m_detected_format;
  std::string m_current_filename;

  // The underlying reader we delegate to
  kwiver::vital::algo::detected_object_set_input_sptr m_reader;

  // Configuration to pass to underlying readers
  kwiver::vital::config_block_sptr m_config;
};


// -----------------------------------------------------------------------------------
std::string
read_detected_object_set_auto::priv
::detect_format( std::string const& filename )
{
  // First, check explicit format extensions
  if( ends_with_ci( filename, ".dive.json" ) )
  {
    return "dive";
  }

  if( ends_with_ci( filename, ".coco.json" ) )
  {
    return "coco";
  }

  // Check general extensions
  std::string ext = to_lower(
    kwiversys::SystemTools::GetFilenameLastExtension( filename ) );

  if( ext == ".csv" )
  {
    return "viame_csv";
  }

  if( ext == ".txt" )
  {
    return "yolo";
  }

  if( ext == ".xml" )
  {
    return "cvat";
  }

  if( ext == ".json" )
  {
    // Need to inspect content to determine JSON format
    std::string content = read_file_header( filename );
    if( content.empty() )
    {
      LOG_WARN( m_parent->logger(),
                "Could not read file for format detection: " << filename );
      return "coco"; // Default to COCO
    }

    return detect_json_format( content );
  }

  // Unknown extension - try to detect by content
  std::string content = read_file_header( filename, 1024 );

  // Check for XML
  if( content.find( "<?xml" ) != std::string::npos ||
      content.find( "<annotations" ) != std::string::npos )
  {
    return "cvat";
  }

  // Check for JSON
  size_t first_nonspace = content.find_first_not_of( " \t\r\n" );
  if( first_nonspace != std::string::npos && content[first_nonspace] == '{' )
  {
    return detect_json_format( content );
  }

  // Check for CSV (has commas and typical VIAME CSV structure)
  if( content.find( ',' ) != std::string::npos )
  {
    // Count commas in first line
    size_t newline = content.find( '\n' );
    std::string first_line = ( newline != std::string::npos )
                             ? content.substr( 0, newline ) : content;
    size_t comma_count = std::count( first_line.begin(), first_line.end(), ',' );

    // VIAME CSV typically has 9+ comma-separated fields
    if( comma_count >= 8 )
    {
      return "viame_csv";
    }
  }

  // Default to YOLO (text file with image paths)
  return "yolo";
}


// ===================================================================================
read_detected_object_set_auto
::read_detected_object_set_auto()
  : d( new read_detected_object_set_auto::priv( this ) )
{
  attach_logger( "viame.core.read_detected_object_set_auto" );
}


read_detected_object_set_auto
::~read_detected_object_set_auto()
{
}


// -----------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
read_detected_object_set_auto
::get_configuration() const
{
  auto config = kwiver::vital::algo::detected_object_set_input::get_configuration();

  // Add configuration options that may be needed by underlying readers

  // YOLO options (all optional - will auto-detect if not specified)
  config->set_value( "yolo:classes_file", "",
    "Path to YOLO classes file. If empty, searches for labels.txt/classes.txt." );
  config->set_value( "yolo:image_width", "0",
    "Image width in pixels. If 0, auto-detects from first image." );
  config->set_value( "yolo:image_height", "0",
    "Image height in pixels. If 0, auto-detects from first image." );

  // CVAT options
  config->set_value( "cvat:default_confidence", "1.0",
    "Default confidence for CVAT detections" );

  return config;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_auto
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_config = config;
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_auto
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_auto
::open( std::string const& filename )
{
  d->m_current_filename = filename;

  // Detect format
  d->m_detected_format = d->detect_format( filename );

  LOG_INFO( logger(), "Auto-detected format '" << d->m_detected_format
            << "' for file: " << filename );

  // Create the appropriate reader
  if( d->m_detected_format == "dive" )
  {
    d->m_reader = std::make_shared< read_detected_object_set_dive >();
  }
  else if( d->m_detected_format == "coco" )
  {
    // COCO reader is in Python, use algorithm factory
    kwiver::vital::algo::detected_object_set_input::set_nested_algo_configuration(
      "reader", d->m_config, d->m_reader );

    if( !d->m_reader )
    {
      // Try to create via factory
      kwiver::vital::algo::detected_object_set_input::get_nested_algo_configuration(
        "reader", d->m_config, d->m_reader );

      if( d->m_config )
      {
        d->m_config->set_value( "reader:type", "coco" );
      }

      kwiver::vital::algo::detected_object_set_input::set_nested_algo_configuration(
        "reader", d->m_config, d->m_reader );
    }

    // If still no reader, throw error
    if( !d->m_reader )
    {
      VITAL_THROW( kwiver::vital::algorithm_configuration_exception,
                   "detected_object_set_input", "coco",
                   "COCO reader not available. Make sure Python support is enabled." );
    }
  }
  else if( d->m_detected_format == "viame_csv" )
  {
    d->m_reader = std::make_shared< read_detected_object_set_viame_csv >();
  }
  else if( d->m_detected_format == "yolo" )
  {
    auto yolo_reader = std::make_shared< read_detected_object_set_yolo >();

    // Configure YOLO reader if we have config
    if( d->m_config )
    {
      auto yolo_config = yolo_reader->get_configuration();

      // Transfer YOLO-specific config
      if( d->m_config->has_value( "yolo:classes_file" ) )
      {
        yolo_config->set_value( "classes_file",
          d->m_config->get_value< std::string >( "yolo:classes_file" ) );
      }
      if( d->m_config->has_value( "yolo:image_width" ) )
      {
        yolo_config->set_value( "image_width",
          d->m_config->get_value< std::string >( "yolo:image_width" ) );
      }
      if( d->m_config->has_value( "yolo:image_height" ) )
      {
        yolo_config->set_value( "image_height",
          d->m_config->get_value< std::string >( "yolo:image_height" ) );
      }

      yolo_reader->set_configuration( yolo_config );
    }

    d->m_reader = yolo_reader;
  }
  else if( d->m_detected_format == "cvat" )
  {
    auto cvat_reader = std::make_shared< read_detected_object_set_cvat >();

    // Configure CVAT reader if we have config
    if( d->m_config )
    {
      auto cvat_config = cvat_reader->get_configuration();

      if( d->m_config->has_value( "cvat:default_confidence" ) )
      {
        cvat_config->set_value( "default_confidence",
          d->m_config->get_value< std::string >( "cvat:default_confidence" ) );
      }

      cvat_reader->set_configuration( cvat_config );
    }

    d->m_reader = cvat_reader;
  }
  else
  {
    VITAL_THROW( kwiver::vital::invalid_data,
                 "Unknown format detected: " + d->m_detected_format );
  }

  // Open the file with the underlying reader
  d->m_reader->open( filename );
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_auto
::close()
{
  if( d->m_reader )
  {
    d->m_reader->close();
    d->m_reader.reset();
  }

  d->m_detected_format.clear();
  d->m_current_filename.clear();
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_auto
::read_set( kwiver::vital::detected_object_set_sptr& set, std::string& image_name )
{
  if( !d->m_reader )
  {
    LOG_ERROR( logger(), "No reader available - was open() called?" );
    return false;
  }

  return d->m_reader->read_set( set, image_name );
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_auto
::new_stream()
{
  if( d->m_reader )
  {
    d->m_reader.reset();
  }

  d->m_detected_format.clear();
}

} // end namespace viame
