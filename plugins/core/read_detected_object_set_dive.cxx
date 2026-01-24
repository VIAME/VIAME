/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_detected_object_set_dive
 */

#include "read_detected_object_set_dive.h"

#include <vital/util/data_stream_reader.h>
#include <vital/exceptions.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/vector.hpp>
#include <vital/internal/cereal/types/map.hpp>
#include <vital/internal/cereal/types/string.hpp>
#include <vital/internal/cereal/types/utility.hpp>

#include <kwiversys/SystemTools.hxx>

#include <map>
#include <memory>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>


namespace viame {

// -----------------------------------------------------------------------------------
// DIVE JSON data structures for cereal parsing
// -----------------------------------------------------------------------------------

struct dive_feature
{
  int frame = 0;
  std::vector< double > bounds;  // [x1, y1, x2, y2]
  bool keyframe = false;
  bool interpolate = false;
  std::vector< double > head;
  std::vector< double > tail;
  double fishLength = 0.0;

  template< class Archive >
  void serialize( Archive& ar )
  {
    try { ar( cereal::make_nvp( "frame", frame ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "bounds", bounds ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "keyframe", keyframe ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "interpolate", interpolate ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "head", head ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "tail", tail ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "fishLength", fishLength ) ); } catch(...) {}
  }
};

struct dive_track
{
  int id = 0;
  int begin = 0;
  int end = 0;
  std::vector< std::pair< std::string, double > > confidencePairs;
  std::vector< dive_feature > features;

  template< class Archive >
  void serialize( Archive& ar )
  {
    try { ar( cereal::make_nvp( "id", id ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "begin", begin ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "end", end ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "confidencePairs", confidencePairs ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "features", features ) ); } catch(...) {}
  }
};

struct dive_annotation_file
{
  std::map< std::string, dive_track > tracks;
  int version = 1;

  template< class Archive >
  void serialize( Archive& ar )
  {
    try { ar( cereal::make_nvp( "tracks", tracks ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "version", version ) ); } catch(...) {}
  }
};

// -----------------------------------------------------------------------------------
class read_detected_object_set_dive::priv
{
public:
  priv( read_detected_object_set_dive* parent )
    : m_parent( parent )
    , m_first( true )
    , m_current_frame( 0 )
    , m_max_frame( -1 )
  { }

  ~priv() { }

  void read_all();
  void parse_json_file( std::string const& filename );
  void parse_dive_json_manual( std::string const& filename, std::string const& content );

  read_detected_object_set_dive* m_parent;
  bool m_first;

  // Current frame index
  int m_current_frame;
  int m_max_frame;

  // Map of detected objects indexed by frame number
  std::map< int, kwiver::vital::detected_object_set_sptr > m_detection_by_frame;

  // Map of frame number to image name (if provided in the input)
  std::map< int, std::string > m_frame_to_image;
};


// ===================================================================================
read_detected_object_set_dive
::read_detected_object_set_dive()
  : d( new read_detected_object_set_dive::priv( this ) )
{
  attach_logger( "viame.core.read_detected_object_set_dive" );
}


read_detected_object_set_dive
::~read_detected_object_set_dive()
{
}


// -----------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
read_detected_object_set_dive
::get_configuration() const
{
  auto config = kwiver::vital::algo::detected_object_set_input::get_configuration();
  return config;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_dive
::set_configuration( kwiver::vital::config_block_sptr config )
{
  // No configuration options currently
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_dive
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------------
bool
read_detected_object_set_dive
::read_set( kwiver::vital::detected_object_set_sptr& set, std::string& image_name )
{
  if( d->m_first )
  {
    // Read in all detections from the JSON file
    d->read_all();
    d->m_first = false;
    d->m_current_frame = 0;
  }

  // Test for end of all frames
  if( d->m_current_frame > d->m_max_frame )
  {
    set = std::make_shared< kwiver::vital::detected_object_set >();
    return false;
  }

  // Return detection set for current frame
  auto itr = d->m_detection_by_frame.find( d->m_current_frame );
  if( itr != d->m_detection_by_frame.end() )
  {
    set = itr->second;
  }
  else
  {
    set = std::make_shared< kwiver::vital::detected_object_set >();
  }

  // Set image name if we have it
  auto name_itr = d->m_frame_to_image.find( d->m_current_frame );
  if( name_itr != d->m_frame_to_image.end() )
  {
    image_name = name_itr->second;
  }

  ++d->m_current_frame;
  return true;
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_dive
::new_stream()
{
  d->m_first = true;
  d->m_detection_by_frame.clear();
  d->m_frame_to_image.clear();
  d->m_current_frame = 0;
  d->m_max_frame = -1;
}


// ===================================================================================
void
read_detected_object_set_dive::priv
::read_all()
{
  m_detection_by_frame.clear();
  m_frame_to_image.clear();
  m_current_frame = 0;
  m_max_frame = -1;

  // Read the JSON filename from the stream (first non-empty, non-comment line)
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( m_parent->stream() );

  while( stream_reader.getline( line ) )
  {
    // Trim whitespace
    size_t start = line.find_first_not_of( " \t\r\n" );
    if( start == std::string::npos )
    {
      continue; // Skip empty lines
    }
    if( line[start] == '#' )
    {
      continue; // Skip comments
    }

    size_t end = line.find_last_not_of( " \t\r\n" );
    std::string json_file = line.substr( start, end - start + 1 );

    // Parse the JSON file
    parse_json_file( json_file );
  }

  LOG_DEBUG( m_parent->logger(),
             "Loaded detections for " << m_detection_by_frame.size()
             << " frames from DIVE JSON" );
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_dive::priv
::parse_json_file( std::string const& filename )
{
  std::ifstream ifs( filename );
  if( !ifs )
  {
    LOG_ERROR( m_parent->logger(),
               "Could not open DIVE JSON file: " << filename );
    return;
  }

  try
  {
    cereal::JSONInputArchive archive( ifs );

    dive_annotation_file dive_data;
    archive( cereal::make_nvp( "root", dive_data ) );

    // Process each track
    for( auto const& track_pair : dive_data.tracks )
    {
      dive_track const& track = track_pair.second;

      // Get the primary class label and confidence from confidencePairs
      std::string primary_label;
      double primary_confidence = 1.0;

      if( !track.confidencePairs.empty() )
      {
        primary_label = track.confidencePairs[0].first;
        primary_confidence = track.confidencePairs[0].second;
      }

      // Process each feature (detection) in the track
      for( dive_feature const& feature : track.features )
      {
        int frame = feature.frame;

        // Update max frame
        if( frame > m_max_frame )
        {
          m_max_frame = frame;
        }

        // Skip features without bounds
        if( feature.bounds.size() < 4 )
        {
          continue;
        }

        // Create bounding box from bounds [x1, y1, x2, y2]
        kwiver::vital::bounding_box_d bbox(
          feature.bounds[0],
          feature.bounds[1],
          feature.bounds[2],
          feature.bounds[3] );

        // Create detected object type with all confidence pairs
        auto dot = std::make_shared< kwiver::vital::detected_object_type >();
        for( auto const& cp : track.confidencePairs )
        {
          dot->set_score( cp.first, cp.second );
        }

        // Create detection
        auto det = std::make_shared< kwiver::vital::detected_object >(
          bbox, primary_confidence, dot );

        // Ensure we have a detection set for this frame
        if( m_detection_by_frame.find( frame ) == m_detection_by_frame.end() )
        {
          m_detection_by_frame[ frame ] =
            std::make_shared< kwiver::vital::detected_object_set >();
        }

        m_detection_by_frame[ frame ]->add( det );
      }
    }
  }
  catch( std::exception const& e )
  {
    LOG_ERROR( m_parent->logger(),
               "Error parsing DIVE JSON file '" << filename << "': " << e.what() );

    // Try alternate parsing approach for compatibility
    ifs.clear();
    ifs.seekg( 0 );

    try
    {
      // Read the entire file content
      std::stringstream buffer;
      buffer << ifs.rdbuf();
      std::string content = buffer.str();

      // Simple JSON parsing for DIVE format
      // Look for "tracks" object and parse track data
      parse_dive_json_manual( filename, content );
    }
    catch( std::exception const& e2 )
    {
      LOG_ERROR( m_parent->logger(),
                 "Fallback parsing also failed: " << e2.what() );
    }
  }
}

// -----------------------------------------------------------------------------------
// Manual JSON parsing for DIVE format as fallback
void
read_detected_object_set_dive::priv
::parse_dive_json_manual( std::string const& filename, std::string const& content )
{
  // This is a simplified manual parser for DIVE JSON format
  // It handles the basic structure without a full JSON library

  // Find "tracks" object
  size_t tracks_pos = content.find( "\"tracks\"" );
  if( tracks_pos == std::string::npos )
  {
    LOG_WARN( m_parent->logger(), "No 'tracks' object found in DIVE JSON" );
    return;
  }

  // Find each track by looking for patterns like "id": <number>
  // and "features": [ ... ]

  size_t pos = tracks_pos;
  while( ( pos = content.find( "\"features\"", pos ) ) != std::string::npos )
  {
    // Find the array start
    size_t array_start = content.find( '[', pos );
    if( array_start == std::string::npos )
    {
      break;
    }

    // Find matching array end (handle nested arrays)
    int bracket_count = 1;
    size_t array_end = array_start + 1;
    while( array_end < content.size() && bracket_count > 0 )
    {
      if( content[array_end] == '[' )
      {
        bracket_count++;
      }
      else if( content[array_end] == ']' )
      {
        bracket_count--;
      }
      array_end++;
    }

    // Look for confidencePairs before this features array
    size_t conf_search_start = ( pos > 500 ) ? pos - 500 : 0;
    std::string search_region = content.substr( conf_search_start, pos - conf_search_start );

    std::string primary_label = "unknown";
    double primary_confidence = 1.0;

    size_t conf_pos = search_region.rfind( "\"confidencePairs\"" );
    if( conf_pos != std::string::npos )
    {
      // Parse first confidence pair
      size_t pair_start = search_region.find( '[', conf_pos );
      if( pair_start != std::string::npos )
      {
        pair_start = search_region.find( '[', pair_start + 1 );
        if( pair_start != std::string::npos )
        {
          size_t label_start = search_region.find( '"', pair_start );
          if( label_start != std::string::npos )
          {
            size_t label_end = search_region.find( '"', label_start + 1 );
            if( label_end != std::string::npos )
            {
              primary_label = search_region.substr( label_start + 1,
                                                     label_end - label_start - 1 );
            }

            // Find confidence value
            size_t comma = search_region.find( ',', label_end );
            if( comma != std::string::npos )
            {
              size_t num_end = search_region.find( ']', comma );
              if( num_end != std::string::npos )
              {
                std::string conf_str = search_region.substr( comma + 1,
                                                              num_end - comma - 1 );
                // Trim whitespace
                size_t ns = conf_str.find_first_not_of( " \t\n\r" );
                size_t ne = conf_str.find_last_not_of( " \t\n\r" );
                if( ns != std::string::npos && ne != std::string::npos )
                {
                  conf_str = conf_str.substr( ns, ne - ns + 1 );
                  try
                  {
                    primary_confidence = std::stod( conf_str );
                  }
                  catch( ... ) {}
                }
              }
            }
          }
        }
      }
    }

    // Parse features in this array
    std::string features_str = content.substr( array_start, array_end - array_start );

    // Find each feature object with "frame" and "bounds"
    size_t feat_pos = 0;
    while( ( feat_pos = features_str.find( "\"frame\"", feat_pos ) ) != std::string::npos )
    {
      // Parse frame number
      size_t colon = features_str.find( ':', feat_pos );
      if( colon == std::string::npos )
      {
        feat_pos++;
        continue;
      }

      size_t num_start = features_str.find_first_of( "0123456789", colon );
      if( num_start == std::string::npos )
      {
        feat_pos++;
        continue;
      }

      size_t num_end = features_str.find_first_not_of( "0123456789", num_start );
      int frame = std::atoi( features_str.substr( num_start,
                                                   num_end - num_start ).c_str() );

      // Update max frame
      if( frame > m_max_frame )
      {
        m_max_frame = frame;
      }

      // Find bounds for this feature
      size_t bounds_pos = features_str.find( "\"bounds\"", feat_pos );
      if( bounds_pos == std::string::npos || bounds_pos > feat_pos + 200 )
      {
        feat_pos = num_end;
        continue;
      }

      size_t bounds_array_start = features_str.find( '[', bounds_pos );
      size_t bounds_array_end = features_str.find( ']', bounds_array_start );

      if( bounds_array_start != std::string::npos &&
          bounds_array_end != std::string::npos )
      {
        std::string bounds_str = features_str.substr(
          bounds_array_start + 1, bounds_array_end - bounds_array_start - 1 );

        // Parse four numbers
        std::vector< double > bounds;
        std::stringstream ss( bounds_str );
        std::string token;
        while( std::getline( ss, token, ',' ) )
        {
          // Trim
          size_t ts = token.find_first_not_of( " \t\n\r" );
          size_t te = token.find_last_not_of( " \t\n\r" );
          if( ts != std::string::npos && te != std::string::npos )
          {
            try
            {
              bounds.push_back( std::stod( token.substr( ts, te - ts + 1 ) ) );
            }
            catch( ... ) {}
          }
        }

        if( bounds.size() >= 4 )
        {
          kwiver::vital::bounding_box_d bbox(
            bounds[0], bounds[1], bounds[2], bounds[3] );

          auto dot = std::make_shared< kwiver::vital::detected_object_type >();
          dot->set_score( primary_label, primary_confidence );

          auto det = std::make_shared< kwiver::vital::detected_object >(
            bbox, primary_confidence, dot );

          if( m_detection_by_frame.find( frame ) == m_detection_by_frame.end() )
          {
            m_detection_by_frame[ frame ] =
              std::make_shared< kwiver::vital::detected_object_set >();
          }

          m_detection_by_frame[ frame ]->add( det );
        }
      }

      feat_pos = num_end;
    }

    pos = array_end;
  }
}

} // end namespace viame
