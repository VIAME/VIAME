/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_detected_object_set_dive and shared DIVE parsing
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
// Internal cereal serialization structures (wrap the exported structs)
// -----------------------------------------------------------------------------------

namespace {

struct dive_feature_serial
{
  dive_feature& ref;

  dive_feature_serial( dive_feature& f ) : ref( f ) {}

  template< class Archive >
  void serialize( Archive& ar )
  {
    try { ar( cereal::make_nvp( "frame", ref.frame ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "bounds", ref.bounds ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "keyframe", ref.keyframe ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "interpolate", ref.interpolate ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "head", ref.head ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "tail", ref.tail ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "fishLength", ref.fishLength ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "attributes", ref.attributes ) ); } catch(...) {}
  }
};

struct dive_track_serial
{
  dive_track& ref;

  dive_track_serial( dive_track& t ) : ref( t ) {}

  template< class Archive >
  void serialize( Archive& ar )
  {
    try { ar( cereal::make_nvp( "id", ref.id ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "begin", ref.begin ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "end", ref.end ) ); } catch(...) {}
    try { ar( cereal::make_nvp( "confidencePairs", ref.confidencePairs ) ); } catch(...) {}
    try
    {
      std::vector< dive_feature_serial > features_serial;
      for( auto& f : ref.features )
      {
        features_serial.push_back( dive_feature_serial( f ) );
      }
      ar( cereal::make_nvp( "features", features_serial ) );
    }
    catch(...) {}
    try { ar( cereal::make_nvp( "attributes", ref.attributes ) ); } catch(...) {}
  }
};

} // anonymous namespace

// ===================================================================================
// Shared DIVE parsing function implementations
// ===================================================================================

// -----------------------------------------------------------------------------------
kwiver::vital::detected_object_sptr
create_detected_object_from_dive(
  dive_feature const& feature,
  std::vector< std::pair< std::string, double > > const& confidence_pairs )
{
  if( feature.bounds.size() < 4 )
  {
    return nullptr;
  }

  // Create bounding box from bounds [x1, y1, x2, y2]
  kwiver::vital::bounding_box_d bbox(
    feature.bounds[0],
    feature.bounds[1],
    feature.bounds[2],
    feature.bounds[3] );

  // Get primary confidence
  double primary_confidence = 1.0;
  if( !confidence_pairs.empty() )
  {
    primary_confidence = confidence_pairs[0].second;
  }

  // Create detected object type with all confidence pairs
  auto dot = std::make_shared< kwiver::vital::detected_object_type >();
  for( auto const& cp : confidence_pairs )
  {
    dot->set_score( cp.first, cp.second );
  }

  // Create detection
  return std::make_shared< kwiver::vital::detected_object >(
    bbox, primary_confidence, dot );
}


// -----------------------------------------------------------------------------------
bool
parse_dive_json_manual( std::string const& content,
                        kwiver::vital::logger_handle_t logger,
                        dive_annotation_file& dive_data )
{
  dive_data.tracks.clear();

  // Find "tracks" object
  size_t tracks_pos = content.find( "\"tracks\"" );
  if( tracks_pos == std::string::npos )
  {
    LOG_WARN( logger, "No 'tracks' object found in DIVE JSON" );
    return false;
  }

  // Find each track by looking for "features" arrays
  size_t pos = tracks_pos;
  int track_counter = 0;

  while( ( pos = content.find( "\"features\"", pos ) ) != std::string::npos )
  {
    dive_track track;
    track.id = track_counter++;

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

    // Look for track ID before this features array
    size_t id_search_start = ( pos > 200 ) ? pos - 200 : 0;
    std::string id_region = content.substr( id_search_start, pos - id_search_start );
    size_t id_pos = id_region.rfind( "\"id\"" );
    if( id_pos != std::string::npos )
    {
      size_t colon = id_region.find( ':', id_pos );
      if( colon != std::string::npos )
      {
        size_t num_start = id_region.find_first_of( "0123456789", colon );
        if( num_start != std::string::npos )
        {
          size_t num_end = id_region.find_first_not_of( "0123456789", num_start );
          track.id = std::atoi( id_region.substr( num_start,
                                                   num_end - num_start ).c_str() );
        }
      }
    }

    // Look for confidencePairs before this features array
    size_t conf_search_start = ( pos > 500 ) ? pos - 500 : 0;
    std::string search_region = content.substr( conf_search_start, pos - conf_search_start );

    size_t conf_pos = search_region.rfind( "\"confidencePairs\"" );
    if( conf_pos != std::string::npos )
    {
      // Parse confidence pairs array [[label, conf], ...]
      size_t pairs_start = search_region.find( '[', conf_pos );
      if( pairs_start != std::string::npos )
      {
        // Find the end of the outer array
        int outer_bracket = 1;
        size_t pairs_end = pairs_start + 1;
        while( pairs_end < search_region.size() && outer_bracket > 0 )
        {
          if( search_region[pairs_end] == '[' )
          {
            outer_bracket++;
          }
          else if( search_region[pairs_end] == ']' )
          {
            outer_bracket--;
          }
          pairs_end++;
        }

        std::string pairs_str = search_region.substr( pairs_start, pairs_end - pairs_start );

        // Find each inner pair [label, conf]
        size_t pair_pos = 0;
        while( ( pair_pos = pairs_str.find( '[', pair_pos + 1 ) ) != std::string::npos )
        {
          size_t pair_end = pairs_str.find( ']', pair_pos );
          if( pair_end == std::string::npos )
          {
            break;
          }

          std::string pair_content = pairs_str.substr( pair_pos + 1, pair_end - pair_pos - 1 );

          // Parse "label", confidence
          size_t label_start = pair_content.find( '"' );
          if( label_start != std::string::npos )
          {
            size_t label_end = pair_content.find( '"', label_start + 1 );
            if( label_end != std::string::npos )
            {
              std::string label = pair_content.substr( label_start + 1,
                                                        label_end - label_start - 1 );

              size_t comma = pair_content.find( ',', label_end );
              if( comma != std::string::npos )
              {
                std::string conf_str = pair_content.substr( comma + 1 );
                size_t ns = conf_str.find_first_not_of( " \t\n\r" );
                size_t ne = conf_str.find_last_not_of( " \t\n\r" );
                if( ns != std::string::npos && ne != std::string::npos )
                {
                  conf_str = conf_str.substr( ns, ne - ns + 1 );
                  try
                  {
                    double confidence = std::stod( conf_str );
                    track.confidencePairs.push_back( std::make_pair( label, confidence ) );
                  }
                  catch( ... ) {}
                }
              }
            }
          }

          pair_pos = pair_end;
        }
      }
    }

    // Parse features in this array
    std::string features_str = content.substr( array_start, array_end - array_start );

    size_t feat_pos = 0;
    while( ( feat_pos = features_str.find( "\"frame\"", feat_pos ) ) != std::string::npos )
    {
      dive_feature feature;

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
      feature.frame = std::atoi( features_str.substr( num_start,
                                                       num_end - num_start ).c_str() );

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

        std::stringstream ss( bounds_str );
        std::string token;
        while( std::getline( ss, token, ',' ) )
        {
          size_t ts = token.find_first_not_of( " \t\n\r" );
          size_t te = token.find_last_not_of( " \t\n\r" );
          if( ts != std::string::npos && te != std::string::npos )
          {
            try
            {
              feature.bounds.push_back( std::stod( token.substr( ts, te - ts + 1 ) ) );
            }
            catch( ... ) {}
          }
        }
      }

      // Check for keyframe
      size_t keyframe_pos = features_str.find( "\"keyframe\"", feat_pos );
      if( keyframe_pos != std::string::npos && keyframe_pos < feat_pos + 300 )
      {
        size_t true_pos = features_str.find( "true", keyframe_pos );
        if( true_pos != std::string::npos && true_pos < keyframe_pos + 20 )
        {
          feature.keyframe = true;
        }
      }

      if( feature.bounds.size() >= 4 )
      {
        track.features.push_back( feature );
      }

      feat_pos = num_end;
    }

    // Update track begin/end from features
    if( !track.features.empty() )
    {
      track.begin = track.features.front().frame;
      track.end = track.features.back().frame;

      for( auto const& f : track.features )
      {
        if( f.frame < track.begin ) track.begin = f.frame;
        if( f.frame > track.end ) track.end = f.frame;
      }

      dive_data.tracks[ std::to_string( track.id ) ] = track;
    }

    pos = array_end;
  }

  return !dive_data.tracks.empty();
}


// -----------------------------------------------------------------------------------
bool
parse_dive_json_file( std::string const& filename,
                      kwiver::vital::logger_handle_t logger,
                      dive_annotation_file& dive_data )
{
  std::ifstream ifs( filename );
  if( !ifs )
  {
    LOG_ERROR( logger, "Could not open DIVE JSON file: " << filename );
    return false;
  }

  try
  {
    cereal::JSONInputArchive archive( ifs );

    // We need custom deserialization since our structs don't have serialize methods
    std::map< std::string, dive_track > tracks_map;

    // Try to parse the root object
    try
    {
      archive( cereal::make_nvp( "version", dive_data.version ) );
    }
    catch( ... ) {}

    // Parse tracks - this is more complex due to cereal limitations
    // Fall back to manual parsing for now
    ifs.clear();
    ifs.seekg( 0 );

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string content = buffer.str();

    return parse_dive_json_manual( content, logger, dive_data );
  }
  catch( std::exception const& e )
  {
    LOG_DEBUG( logger, "Cereal parsing failed: " << e.what()
               << ", trying manual parser" );

    ifs.clear();
    ifs.seekg( 0 );

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string content = buffer.str();

    return parse_dive_json_manual( content, logger, dive_data );
  }
}


// ===================================================================================
// Detection reader implementation
// ===================================================================================

// -----------------------------------------------------------------------------------
class read_detected_object_set_dive::priv
{
public:
  priv( read_detected_object_set_dive& parent )
    : m_parent( &parent )
    , m_first( true )
    , m_current_frame( 0 )
    , m_max_frame( -1 )
  { }

  ~priv() { }

  void read_all();

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
::~read_detected_object_set_dive()
{
}


// -----------------------------------------------------------------------------------
void
read_detected_object_set_dive
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.core.read_detected_object_set_dive" );
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

    // Parse the JSON file using shared function
    dive_annotation_file dive_data;
    if( !parse_dive_json_file( json_file, m_parent->logger(), dive_data ) )
    {
      LOG_ERROR( m_parent->logger(),
                 "Failed to parse DIVE JSON file: " << json_file );
      continue;
    }

    // Process each track
    for( auto const& track_pair : dive_data.tracks )
    {
      dive_track const& track = track_pair.second;

      // Process each feature (detection) in the track
      for( dive_feature const& feature : track.features )
      {
        int frame = feature.frame;

        // Update max frame
        if( frame > m_max_frame )
        {
          m_max_frame = frame;
        }

        // Create detection using shared function
        auto det = create_detected_object_from_dive( feature, track.confidencePairs );
        if( !det )
        {
          continue;
        }

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

  LOG_DEBUG( m_parent->logger(),
             "Loaded detections for " << m_detection_by_frame.size()
             << " frames from DIVE JSON" );
}

} // end namespace viame
