/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_object_track_set_auto
 */

#include "read_object_track_set_auto.h"

#include "read_object_track_set_dive.h"
#include "read_object_track_set_viame_csv.h"
#include "utilities_file.h"

#include <vital/algo/read_object_track_set.h>
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
std::string read_file_header_track( std::string const& filename, size_t max_bytes = 4096 )
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

// Detect if JSON content is DIVE format (has tracks with features)
bool is_dive_json_format( std::string const& content )
{
  // DIVE format has "tracks" as a top-level key with track objects containing "features"
  bool has_tracks = content.find( "\"tracks\"" ) != std::string::npos;
  bool has_features = content.find( "\"features\"" ) != std::string::npos;
  bool has_confidence_pairs = content.find( "\"confidencePairs\"" ) != std::string::npos;

  return has_tracks && ( has_features || has_confidence_pairs );
}

} // anonymous namespace


// -----------------------------------------------------------------------------------
class read_object_track_set_auto::priv
{
public:
  priv( read_object_track_set_auto* parent )
    : m_parent( parent )
    , m_detected_format( "" )
  { }

  ~priv() { }

  std::string detect_format( std::string const& filename );

  read_object_track_set_auto* m_parent;
  std::string m_detected_format;
  std::string m_current_filename;

  // The underlying reader we delegate to
  kwiver::vital::algo::read_object_track_set_sptr m_reader;

  // Configuration to pass to underlying readers
  kwiver::vital::config_block_sptr m_config;
};


// -----------------------------------------------------------------------------------
std::string
read_object_track_set_auto::priv
::detect_format( std::string const& filename )
{
  // First, check explicit format extensions
  if( ends_with_ci( filename, ".dive.json" ) )
  {
    return "dive";
  }

  // Check general extensions
  std::string ext = to_lower(
    kwiversys::SystemTools::GetFilenameLastExtension( filename ) );

  if( ext == ".csv" )
  {
    return "viame_csv";
  }

  if( ext == ".json" )
  {
    // Need to inspect content to determine if it's DIVE format
    std::string content = read_file_header_track( filename );
    if( content.empty() )
    {
      LOG_WARN( m_parent->logger(),
                "Could not read file for format detection: " << filename );
      return "dive"; // Default to DIVE for JSON
    }

    if( is_dive_json_format( content ) )
    {
      return "dive";
    }

    // Unknown JSON format - try DIVE anyway
    return "dive";
  }

  // Unknown extension - try to detect by content
  std::string content = read_file_header_track( filename, 1024 );

  // Check for JSON (DIVE)
  size_t first_nonspace = content.find_first_not_of( " \t\r\n" );
  if( first_nonspace != std::string::npos && content[first_nonspace] == '{' )
  {
    if( is_dive_json_format( content ) )
    {
      return "dive";
    }
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

  // Default to VIAME CSV
  return "viame_csv";
}


// ===================================================================================
read_object_track_set_auto
::read_object_track_set_auto()
  : d( new read_object_track_set_auto::priv( this ) )
{
  attach_logger( "viame.core.read_object_track_set_auto" );
}


read_object_track_set_auto
::~read_object_track_set_auto()
{
}


// -----------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
read_object_track_set_auto
::get_configuration() const
{
  auto config = kwiver::vital::algo::read_object_track_set::get_configuration();

  // Add configuration options that may be needed by underlying readers

  // VIAME CSV options
  config->set_value( "viame_csv:delimiter", ",",
    "Delimiter character for CSV parsing" );
  config->set_value( "viame_csv:batch_load", "false",
    "Load all tracks at once (true) or stream frame-by-frame (false)" );

  // DIVE options
  config->set_value( "dive:batch_load", "true",
    "Load all tracks at once (true) or stream frame-by-frame (false)" );

  return config;
}


// -----------------------------------------------------------------------------------
void
read_object_track_set_auto
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_config = config;
}


// -----------------------------------------------------------------------------------
bool
read_object_track_set_auto
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------------
void
read_object_track_set_auto
::open( std::string const& filename )
{
  d->m_current_filename = filename;

  // Detect format
  d->m_detected_format = d->detect_format( filename );

  LOG_INFO( logger(), "Auto-detected track format '" << d->m_detected_format
            << "' for file: " << filename );

  // Create the appropriate reader
  if( d->m_detected_format == "dive" )
  {
    auto dive_reader = std::make_shared< read_object_track_set_dive >();

    // Configure DIVE reader if we have config
    if( d->m_config )
    {
      auto dive_config = dive_reader->get_configuration();

      if( d->m_config->has_value( "dive:batch_load" ) )
      {
        dive_config->set_value( "batch_load",
          d->m_config->get_value< std::string >( "dive:batch_load" ) );
      }

      dive_reader->set_configuration( dive_config );
    }

    d->m_reader = dive_reader;
  }
  else if( d->m_detected_format == "viame_csv" )
  {
    auto csv_reader = std::make_shared< read_object_track_set_viame_csv >();

    // Configure VIAME CSV reader if we have config
    if( d->m_config )
    {
      auto csv_config = csv_reader->get_configuration();

      if( d->m_config->has_value( "viame_csv:delimiter" ) )
      {
        csv_config->set_value( "delimiter",
          d->m_config->get_value< std::string >( "viame_csv:delimiter" ) );
      }
      if( d->m_config->has_value( "viame_csv:batch_load" ) )
      {
        csv_config->set_value( "batch_load",
          d->m_config->get_value< std::string >( "viame_csv:batch_load" ) );
      }

      csv_reader->set_configuration( csv_config );
    }

    d->m_reader = csv_reader;
  }
  else
  {
    VITAL_THROW( kwiver::vital::invalid_data,
                 "Unknown track format detected: " + d->m_detected_format );
  }

  // Open the file with the underlying reader
  d->m_reader->open( filename );
}


// -----------------------------------------------------------------------------------
void
read_object_track_set_auto
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
read_object_track_set_auto
::read_set( kwiver::vital::object_track_set_sptr& set )
{
  if( !d->m_reader )
  {
    LOG_ERROR( logger(), "No reader available - was open() called?" );
    return false;
  }

  return d->m_reader->read_set( set );
}

} // end namespace viame
