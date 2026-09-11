/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of the DIVE JSON track writer and shared serializer
 */

#include "write_object_track_set_dive.h"
#include "utilities_target_clfr.h"

#include <vital/types/object_track_set.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/attribute_set.h>
#include <vital/logger/logger.h>

#include <vital/internal/cereal/external/rapidjson/document.h>
#include <vital/internal/cereal/external/rapidjson/prettywriter.h>
#include <vital/internal/cereal/external/rapidjson/writer.h>
#include <vital/internal/cereal/external/rapidjson/ostreamwrapper.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <string>

namespace viame {

namespace kv = kwiver::vital;

namespace {

using json_value = rapidjson::Value;
using json_alloc = rapidjson::Document::AllocatorType;

// -----------------------------------------------------------------------------
json_value
json_string( std::string const& text, json_alloc& alloc )
{
  return json_value( text.c_str(), static_cast< rapidjson::SizeType >( text.size() ), alloc );
}

// -----------------------------------------------------------------------------
// Attribute values as typed JSON: numbers and booleans stay typed, the rest
// are strings, mirroring how DIVE deduces types from CSV attributes.
json_value
typed_json( std::string const& text, json_alloc& alloc )
{
  if( text == "true" ) { return json_value( true ); }
  if( text == "false" ) { return json_value( false ); }

  if( !text.empty() )
  {
    char* end = nullptr;
    const double number = std::strtod( text.c_str(), &end );
    if( end && *end == '\0' && std::isfinite( number ) )
    {
      const double rounded = std::floor( number );
      if( rounded == number && std::fabs( number ) < 1e15 &&
          text.find_first_of( ".eE" ) == std::string::npos )
      {
        return json_value( static_cast< int64_t >( rounded ) );
      }
      return json_value( number );
    }
  }
  return json_string( text, alloc );
}

// -----------------------------------------------------------------------------
json_value
any_to_json( kv::any const& value, json_alloc& alloc )
{
  if( value.type() == typeid( std::string ) )
  {
    return json_string( kv::any_cast< std::string >( value ), alloc );
  }
  if( value.type() == typeid( double ) )
  {
    return json_value( kv::any_cast< double >( value ) );
  }
  if( value.type() == typeid( float ) )
  {
    return json_value( static_cast< double >( kv::any_cast< float >( value ) ) );
  }
  if( value.type() == typeid( int ) )
  {
    return json_value( kv::any_cast< int >( value ) );
  }
  if( value.type() == typeid( unsigned ) )
  {
    return json_value( kv::any_cast< unsigned >( value ) );
  }
  if( value.type() == typeid( int64_t ) )
  {
    return json_value( kv::any_cast< int64_t >( value ) );
  }
  if( value.type() == typeid( bool ) )
  {
    return json_value( kv::any_cast< bool >( value ) );
  }
  return json_value( rapidjson::kNullType );
}

// -----------------------------------------------------------------------------
json_value
point_json( double x, double y, json_alloc& alloc )
{
  json_value point( rapidjson::kArrayType );
  point.PushBack( json_value( x ), alloc );
  point.PushBack( json_value( y ), alloc );
  return point;
}

// -----------------------------------------------------------------------------
json_value
geojson_feature( char const* type, std::string const& key,
                 json_value& coordinates, json_alloc& alloc )
{
  json_value properties( rapidjson::kObjectType );
  properties.AddMember( "key", json_string( key, alloc ), alloc );

  json_value geometry( rapidjson::kObjectType );
  geometry.AddMember( "type", json_value( rapidjson::StringRef( type ) ), alloc );
  geometry.AddMember( "coordinates", coordinates, alloc );

  json_value feature( rapidjson::kObjectType );
  feature.AddMember( "type", "Feature", alloc );
  feature.AddMember( "properties", properties, alloc );
  feature.AddMember( "geometry", geometry, alloc );
  return feature;
}

// -----------------------------------------------------------------------------
// Split a detection note of the VIAME ":key=value" form, as produced by the
// CSV reader for "(atr)" cells. Returns false for free-form notes.
bool
split_attribute_note( std::string const& note, std::string& key, std::string& value )
{
  if( note.size() < 3 || note[0] != ':' )
  {
    return false;
  }
  const std::size_t equals = note.find( '=' );
  if( equals == std::string::npos || equals == 1 )
  {
    return false;
  }
  key = note.substr( 1, equals - 1 );
  value = note.substr( equals + 1 );
  return true;
}

// -----------------------------------------------------------------------------
json_value
feature_json( kv::object_track_state const& state,
              dive_write_options const& options, json_alloc& alloc )
{
  json_value feature( rapidjson::kObjectType );

  feature.AddMember( "frame",
    json_value( static_cast< int64_t >( state.frame() + options.frame_id_adjustment ) ),
    alloc );
  feature.AddMember( "keyframe", json_value( true ), alloc );
  feature.AddMember( "interpolate", json_value( false ), alloc );

  kv::detected_object_scptr det = state.detection();
  if( !det )
  {
    return feature;
  }

  const kv::bounding_box_d bbox = det->bounding_box();
  json_value bounds( rapidjson::kArrayType );
  bounds.PushBack( json_value( bbox.min_x() ), alloc );
  bounds.PushBack( json_value( bbox.min_y() ), alloc );
  bounds.PushBack( json_value( bbox.max_x() ), alloc );
  bounds.PushBack( json_value( bbox.max_y() ), alloc );
  feature.AddMember( "bounds", bounds, alloc );

  json_value geometry_features( rapidjson::kArrayType );

  // Head and tail keypoints: written both as top-level fields and as the
  // point / line features the DIVE importer produces from VIAME CSV.
  const auto keypoints = det->keypoints();
  const auto head = keypoints.find( "head" );
  const auto tail = keypoints.find( "tail" );

  if( head != keypoints.end() )
  {
    feature.AddMember( "head", point_json( head->second[0], head->second[1], alloc ), alloc );
    json_value coords = point_json( head->second[0], head->second[1], alloc );
    geometry_features.PushBack( geojson_feature( "Point", "head", coords, alloc ), alloc );
  }
  if( tail != keypoints.end() )
  {
    feature.AddMember( "tail", point_json( tail->second[0], tail->second[1], alloc ), alloc );
    json_value coords = point_json( tail->second[0], tail->second[1], alloc );
    geometry_features.PushBack( geojson_feature( "Point", "tail", coords, alloc ), alloc );
  }
  if( head != keypoints.end() && tail != keypoints.end() )
  {
    json_value line( rapidjson::kArrayType );
    line.PushBack( point_json( head->second[0], head->second[1], alloc ), alloc );
    line.PushBack( point_json( tail->second[0], tail->second[1], alloc ), alloc );
    geometry_features.PushBack( geojson_feature( "LineString", "HeadTails", line, alloc ), alloc );
  }

  // Explicit polygon outline as a closed GeoJSON polygon ring
  const auto polygon = det->polygon();
  if( !polygon.empty() )
  {
    json_value ring( rapidjson::kArrayType );
    for( auto const& p : polygon )
    {
      ring.PushBack( point_json( p[0], p[1], alloc ), alloc );
    }
    ring.PushBack( point_json( polygon.front()[0], polygon.front()[1], alloc ), alloc );
    json_value rings( rapidjson::kArrayType );
    rings.PushBack( ring, alloc );
    geometry_features.PushBack( geojson_feature( "Polygon", "", rings, alloc ), alloc );
  }

  if( !geometry_features.Empty() )
  {
    json_value collection( rapidjson::kObjectType );
    collection.AddMember( "type", "FeatureCollection", alloc );
    collection.AddMember( "features", geometry_features, alloc );
    feature.AddMember( "geometry", collection, alloc );
  }

  if( det->has_attribute( "length" ) )
  {
    try
    {
      const double length = det->get_attribute< double >( "length" );
      if( length > 0 )
      {
        feature.AddMember( "fishLength", json_value( length ), alloc );
      }
    }
    catch( ... ) {}
  }

  // ":key=value" notes become attributes, anything else stays a note
  json_value attributes( rapidjson::kObjectType );
  json_value notes( rapidjson::kArrayType );
  for( auto const& note : det->notes() )
  {
    std::string key, value;
    if( split_attribute_note( note, key, value ) )
    {
      attributes.AddMember( json_string( key, alloc ), typed_json( value, alloc ), alloc );
    }
    else if( !note.empty() )
    {
      notes.PushBack( json_string( note, alloc ), alloc );
    }
  }
  if( !attributes.ObjectEmpty() )
  {
    feature.AddMember( "attributes", attributes, alloc );
  }
  if( !notes.Empty() )
  {
    feature.AddMember( "notes", notes, alloc );
  }

  return feature;
}

// -----------------------------------------------------------------------------
// Class scores averaged over the states that carry a classification
json_value
confidence_pairs_json( kv::track_sptr const& track,
                       dive_write_options const& options, json_alloc& alloc )
{
  std::map< std::string, double > sums;
  unsigned classified = 0;

  for( auto const& state_ptr : *track )
  {
    auto const* state = dynamic_cast< kv::object_track_state const* >( state_ptr.get() );
    if( !state || !state->detection() || !state->detection()->type() )
    {
      continue;
    }
    ++classified;
    for( auto const& name : state->detection()->type()->class_names() )
    {
      sums[ name ] += state->detection()->type()->score( name );
    }
  }

  std::vector< std::pair< std::string, double > > pairs;
  for( auto const& item : sums )
  {
    pairs.emplace_back( item.first, item.second / classified );
  }
  std::stable_sort( pairs.begin(), pairs.end(),
    []( auto const& a, auto const& b ){ return a.second > b.second; } );

  if( options.top_n_classes > 0 && pairs.size() > options.top_n_classes )
  {
    pairs.resize( options.top_n_classes );
  }

  json_value out( rapidjson::kArrayType );
  for( auto const& pair : pairs )
  {
    json_value entry( rapidjson::kArrayType );
    entry.PushBack( json_string( pair.first, alloc ), alloc );
    entry.PushBack( json_value( pair.second ), alloc );
    out.PushBack( entry, alloc );
  }
  return out;
}

} // anonymous namespace

// =============================================================================
void
write_dive_json( std::ostream& stream,
                 std::vector< kv::track_sptr > const& tracks,
                 dive_write_options const& options )
{
  rapidjson::Document doc( rapidjson::kObjectType );
  json_alloc& alloc = doc.GetAllocator();

  json_value tracks_json( rapidjson::kObjectType );

  for( auto const& track : tracks )
  {
    if( !track || track->empty() )
    {
      continue;
    }

    json_value features( rapidjson::kArrayType );
    int64_t begin = std::numeric_limits< int64_t >::max();
    int64_t end = std::numeric_limits< int64_t >::min();

    for( auto const& state_ptr : *track )
    {
      auto const* state = dynamic_cast< kv::object_track_state const* >( state_ptr.get() );
      if( !state )
      {
        continue;
      }
      const int64_t frame = state->frame() + options.frame_id_adjustment;
      begin = std::min( begin, frame );
      end = std::max( end, frame );
      features.PushBack( feature_json( *state, options, alloc ), alloc );
    }

    if( features.Empty() )
    {
      continue;
    }

    json_value attributes( rapidjson::kObjectType );
    if( auto attrs = track->attributes() )
    {
      for( auto itr = attrs->begin(); itr != attrs->end(); ++itr )
      {
        if( itr->second )
        {
          attributes.AddMember( json_string( itr->first, alloc ),
                                any_to_json( *itr->second, alloc ), alloc );
        }
      }
    }

    json_value entry( rapidjson::kObjectType );
    entry.AddMember( "id", json_value( static_cast< int64_t >( track->id() ) ), alloc );
    entry.AddMember( "meta", json_value( rapidjson::kObjectType ), alloc );
    entry.AddMember( "attributes", attributes, alloc );
    entry.AddMember( "confidencePairs", confidence_pairs_json( track, options, alloc ), alloc );
    entry.AddMember( "features", features, alloc );
    entry.AddMember( "begin", json_value( begin ), alloc );
    entry.AddMember( "end", json_value( end ), alloc );

    tracks_json.AddMember( json_string( std::to_string( track->id() ), alloc ), entry, alloc );
  }

  doc.AddMember( "version", json_value( 2 ), alloc );
  doc.AddMember( "tracks", tracks_json, alloc );
  doc.AddMember( "groups", json_value( rapidjson::kObjectType ), alloc );

  rapidjson::OStreamWrapper wrapper( stream );
  if( options.pretty_print )
  {
    rapidjson::PrettyWriter< rapidjson::OStreamWrapper > writer( wrapper );
    writer.SetIndent( ' ', 2 );
    doc.Accept( writer );
  }
  else
  {
    rapidjson::Writer< rapidjson::OStreamWrapper > writer( wrapper );
    doc.Accept( writer );
  }
  stream << "\n";
  stream.flush();
}

// =============================================================================
void
write_object_track_set_dive
::initialize()
{
  attach_logger( "viame.core.write_object_track_set_dive" );
}

// -----------------------------------------------------------------------------
bool
write_object_track_set_dive
::check_configuration( kv::config_block_sptr ) const
{
  return true;
}

// -----------------------------------------------------------------------------
void
write_object_track_set_dive
::write_set( const kv::object_track_set_sptr& set,
             const kv::timestamp&,
             const std::string& )
{
  if( !set )
  {
    return;
  }
  for( auto const& track : set->tracks() )
  {
    if( track )
    {
      m_tracks[ track->id() ] = track;
    }
  }
}

// -----------------------------------------------------------------------------
void
write_object_track_set_dive
::close()
{
  std::vector< kv::track_sptr > tracks;
  tracks.reserve( m_tracks.size() );
  for( auto const& item : m_tracks )
  {
    tracks.push_back( item.second );
  }

  dive_write_options options;
  options.frame_id_adjustment = c_frame_id_adjustment;
  options.top_n_classes = c_top_n_classes;
  options.pretty_print = c_pretty_print;

  write_dive_json( stream(), tracks, options );
  m_tracks.clear();

  write_object_track_set::close();
}

} // end namespace viame
