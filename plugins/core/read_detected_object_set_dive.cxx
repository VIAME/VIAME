/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for read_detected_object_set_dive and shared DIVE parsing
 */

#include "read_detected_object_set_dive.h"

#include <viame/algorithm_framework/util/data_stream_reader.h>
#include <viame/algorithm_framework/exceptions.h>

// Upstream reads through the rapidjson that cereal ships, reached there as
// `vital/internal/cereal/...`. Phase 5 vendored cereal whole into
// `third_party/cereal` and puts its `external` directory on the include
// path, so the same headers are `<rapidjson/...>` here.
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <kwiversys/SystemTools.hxx>

#include <map>
#include <memory>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>


namespace viame {


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
  auto det = std::make_shared< kwiver::vital::detected_object >(
    bbox, primary_confidence, dot );

  if( feature.head.size() >= 2 )
  {
    det->add_keypoint( "head", { feature.head[0], feature.head[1] } );
  }
  if( feature.tail.size() >= 2 )
  {
    det->add_keypoint( "tail", { feature.tail[0], feature.tail[1] } );
  }
  if( feature.fishLength > 0.0 )
  {
    det->set_attribute( "length", feature.fishLength );
  }
  if( !feature.polygon.empty() )
  {
    std::vector< kwiver::vital::vector_2d > polygon;
    for( auto const& p : feature.polygon )
    {
      polygon.emplace_back( p.first, p.second );
    }
    det->set_polygon( polygon );
  }
  // Attributes travel as ":key=value" notes, the form the CSV writer emits
  for( auto const& attr : feature.attributes )
  {
    det->add_note( ":" + attr.first + "=" + attr.second );
  }
  for( auto const& note : feature.notes )
  {
    det->add_note( note );
  }
  return det;
}

// -----------------------------------------------------------------------------------
// The file handed to open() is either a DIVE JSON document itself or, for
// older configurations, a list of DIVE JSON paths, one per line.
std::vector< std::string >
dive_json_files_from_stream( std::istream& stream, std::string const& filename )
{
  std::vector< std::string > files;
  std::string line;
  kwiver::vital::data_stream_reader stream_reader( stream );

  while( stream_reader.getline( line ) )
  {
    size_t start = line.find_first_not_of( " \t\r\n" );
    if( start == std::string::npos || line[start] == '#' )
    {
      continue;
    }
    if( line[start] == '{' || line[start] == '[' )
    {
      return { filename };
    }
    size_t end = line.find_last_not_of( " \t\r\n" );
    files.push_back( line.substr( start, end - start + 1 ) );
  }
  return files;
}


// -----------------------------------------------------------------------------------

namespace {

// -----------------------------------------------------------------------------------
std::string
json_scalar_to_string( rapidjson::Value const& value )
{
  if( value.IsString() ) { return value.GetString(); }
  if( value.IsBool() ) { return value.GetBool() ? "true" : "false"; }
  if( value.IsInt64() ) { return std::to_string( value.GetInt64() ); }
  if( value.IsNumber() )
  {
    std::ostringstream out;
    out << std::setprecision( 12 ) << value.GetDouble();
    return out.str();
  }
  return {};
}

// -----------------------------------------------------------------------------------
std::vector< double >
json_number_array( rapidjson::Value const& value )
{
  std::vector< double > out;
  if( value.IsArray() )
  {
    for( auto const& item : value.GetArray() )
    {
      if( item.IsNumber() )
      {
        out.push_back( item.GetDouble() );
      }
    }
  }
  return out;
}

// -----------------------------------------------------------------------------------
void
parse_attributes( rapidjson::Value const& object,
                  std::map< std::string, std::string >& attributes )
{
  if( !object.IsObject() )
  {
    return;
  }
  for( auto itr = object.MemberBegin(); itr != object.MemberEnd(); ++itr )
  {
    if( itr->value.IsString() || itr->value.IsNumber() || itr->value.IsBool() )
    {
      attributes[ itr->name.GetString() ] = json_scalar_to_string( itr->value );
    }
  }
}

// -----------------------------------------------------------------------------------
// GeoJSON features inside a DIVE feature: the first polygon's outer ring,
// plus head/tail points when the top-level fields are absent
void
parse_geometry( rapidjson::Value const& geometry, dive_feature& feature )
{
  if( !geometry.IsObject() || !geometry.HasMember( "features" ) ||
      !geometry[ "features" ].IsArray() )
  {
    return;
  }

  for( auto const& item : geometry[ "features" ].GetArray() )
  {
    if( !item.IsObject() || !item.HasMember( "geometry" ) ||
        !item[ "geometry" ].IsObject() )
    {
      continue;
    }
    rapidjson::Value const& shape = item[ "geometry" ];
    if( !shape.HasMember( "type" ) || !shape[ "type" ].IsString() ||
        !shape.HasMember( "coordinates" ) )
    {
      continue;
    }
    const std::string type = shape[ "type" ].GetString();
    std::string key;
    if( item.HasMember( "properties" ) && item[ "properties" ].IsObject() &&
        item[ "properties" ].HasMember( "key" ) &&
        item[ "properties" ][ "key" ].IsString() )
    {
      key = item[ "properties" ][ "key" ].GetString();
    }

    if( type == "Polygon" && feature.polygon.empty() &&
        shape[ "coordinates" ].IsArray() && !shape[ "coordinates" ].Empty() )
    {
      for( auto const& point : shape[ "coordinates" ][ 0 ].GetArray() )
      {
        const auto xy = json_number_array( point );
        if( xy.size() >= 2 )
        {
          feature.polygon.emplace_back( xy[0], xy[1] );
        }
      }
      if( feature.polygon.size() > 1 &&
          feature.polygon.front() == feature.polygon.back() )
      {
        feature.polygon.pop_back();
      }
    }
    else if( type == "Point" && ( key == "head" || key == "tail" ) )
    {
      const auto xy = json_number_array( shape[ "coordinates" ] );
      if( xy.size() >= 2 )
      {
        auto& target = ( key == "head" ) ? feature.head : feature.tail;
        if( target.size() < 2 )
        {
          target = { xy[0], xy[1] };
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------------
bool
parse_feature( rapidjson::Value const& value, dive_feature& feature )
{
  if( !value.IsObject() || !value.HasMember( "frame" ) ||
      !value[ "frame" ].IsNumber() )
  {
    return false;
  }
  feature.frame = value[ "frame" ].GetInt();

  if( value.HasMember( "bounds" ) )
  {
    feature.bounds = json_number_array( value[ "bounds" ] );
  }
  if( value.HasMember( "keyframe" ) && value[ "keyframe" ].IsBool() )
  {
    feature.keyframe = value[ "keyframe" ].GetBool();
  }
  if( value.HasMember( "interpolate" ) && value[ "interpolate" ].IsBool() )
  {
    feature.interpolate = value[ "interpolate" ].GetBool();
  }
  if( value.HasMember( "head" ) )
  {
    feature.head = json_number_array( value[ "head" ] );
  }
  if( value.HasMember( "tail" ) )
  {
    feature.tail = json_number_array( value[ "tail" ] );
  }
  if( value.HasMember( "fishLength" ) && value[ "fishLength" ].IsNumber() )
  {
    feature.fishLength = value[ "fishLength" ].GetDouble();
  }
  if( value.HasMember( "attributes" ) )
  {
    parse_attributes( value[ "attributes" ], feature.attributes );
  }
  if( value.HasMember( "notes" ) && value[ "notes" ].IsArray() )
  {
    for( auto const& note : value[ "notes" ].GetArray() )
    {
      if( note.IsString() )
      {
        feature.notes.push_back( note.GetString() );
      }
    }
  }
  if( value.HasMember( "geometry" ) )
  {
    parse_geometry( value[ "geometry" ], feature );
  }
  return true;
}

// -----------------------------------------------------------------------------------
bool
parse_track( std::string const& key, rapidjson::Value const& value, dive_track& track )
{
  if( !value.IsObject() )
  {
    return false;
  }

  // Version 2 files carry "id"; version 1 files carried "trackId"
  if( value.HasMember( "id" ) && value[ "id" ].IsNumber() )
  {
    track.id = value[ "id" ].GetInt();
  }
  else if( value.HasMember( "trackId" ) && value[ "trackId" ].IsNumber() )
  {
    track.id = value[ "trackId" ].GetInt();
  }
  else
  {
    try { track.id = std::stoi( key ); } catch( ... ) { return false; }
  }

  if( value.HasMember( "confidencePairs" ) && value[ "confidencePairs" ].IsArray() )
  {
    for( auto const& pair : value[ "confidencePairs" ].GetArray() )
    {
      if( pair.IsArray() && pair.Size() >= 2 && pair[0].IsString() && pair[1].IsNumber() )
      {
        track.confidencePairs.emplace_back( pair[0].GetString(), pair[1].GetDouble() );
      }
    }
  }
  if( value.HasMember( "attributes" ) )
  {
    parse_attributes( value[ "attributes" ], track.attributes );
  }
  if( value.HasMember( "features" ) && value[ "features" ].IsArray() )
  {
    for( auto const& item : value[ "features" ].GetArray() )
    {
      dive_feature feature;
      if( parse_feature( item, feature ) )
      {
        track.features.push_back( feature );
      }
    }
  }

  std::sort( track.features.begin(), track.features.end(),
    []( dive_feature const& a, dive_feature const& b ){ return a.frame < b.frame; } );

  if( !track.features.empty() )
  {
    track.begin = track.features.front().frame;
    track.end = track.features.back().frame;
  }
  return !track.features.empty();
}

// -----------------------------------------------------------------------------------
bool
parse_dive_document( rapidjson::Document const& doc,
                     kwiver::vital::logger_handle_t logger,
                     dive_annotation_file& dive_data )
{
  dive_data.tracks.clear();
  dive_data.version = 1;

  if( !doc.IsObject() )
  {
    LOG_ERROR( logger, "DIVE JSON root is not an object" );
    return false;
  }

  if( doc.HasMember( "version" ) && doc[ "version" ].IsNumber() )
  {
    dive_data.version = doc[ "version" ].GetInt();
  }

  // Version 2 keeps the tracks under "tracks"; version 1 had them at the root
  rapidjson::Value const* tracks = &doc;
  if( doc.HasMember( "tracks" ) && doc[ "tracks" ].IsObject() )
  {
    tracks = &doc[ "tracks" ];
  }

  for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
  {
    dive_track track;
    if( parse_track( itr->name.GetString(), itr->value, track ) )
    {
      dive_data.tracks[ itr->name.GetString() ] = track;
    }
  }
  return true;
}

} // anonymous namespace

// -----------------------------------------------------------------------------------
bool
parse_dive_json_manual( std::string const& content,
                        kwiver::vital::logger_handle_t logger,
                        dive_annotation_file& dive_data )
{
  rapidjson::Document doc;
  doc.Parse( content.c_str() );
  if( doc.HasParseError() )
  {
    LOG_ERROR( logger, "DIVE JSON parse error at offset " << doc.GetErrorOffset()
                       << ": " << rapidjson::GetParseError_En( doc.GetParseError() ) );
    return false;
  }
  return parse_dive_document( doc, logger, dive_data );
}

// -----------------------------------------------------------------------------------
bool
parse_dive_json_file( std::string const& filename,
                      kwiver::vital::logger_handle_t logger,
                      dive_annotation_file& dive_data )
{
  std::ifstream ifs( filename, std::ios::binary );
  if( !ifs )
  {
    LOG_ERROR( logger, "Could not open DIVE JSON file: " << filename );
    return false;
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return parse_dive_json_manual( buffer.str(), logger, dive_data );
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
  std::string m_filename;
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
void
read_detected_object_set_dive
::open( std::string const& filename )
{
  kwiver::vital::algo::detected_object_set_input::open( filename );
  d->m_filename = filename;
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

  for( auto const& json_file :
       dive_json_files_from_stream( m_parent->stream(), m_filename ) )
  {
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
