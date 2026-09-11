/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Applet for filtering and analyzing DIVE and COCO JSON files

#include "json.h"
#include <atomic_output.h>

#include <utilities_file.h>

// Pulled in for the vendored rapidjson headers and to route rapidjson
// assertions to exceptions rather than aborts
#include <cereal/archives/json.hpp>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>

#include <viame/algorithm_framework/logger/logger.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kv = kwiver::vital;

namespace viame {
namespace tools {

namespace {

using json_doc = rapidjson::Document;
using json_value = rapidjson::Value;
using json_alloc = json_doc::AllocatorType;
using json_size = rapidjson::SizeType;

// =======================================================================================
// JSON value access helpers

const json_value*
find( const json_value& value, const char* name )
{
  if( !value.IsObject() )
  {
    return nullptr;
  }

  auto itr = value.FindMember( name );
  return itr != value.MemberEnd() ? &itr->value : nullptr;
}

json_value*
find( json_value& value, const char* name )
{
  if( !value.IsObject() )
  {
    return nullptr;
  }

  auto itr = value.FindMember( name );
  return itr != value.MemberEnd() ? &itr->value : nullptr;
}

int
json_int( const json_value& value )
{
  if( value.IsInt() )
  {
    return value.GetInt();
  }
  if( value.IsInt64() )
  {
    return static_cast< int >( value.GetInt64() );
  }
  if( value.IsUint() )
  {
    return static_cast< int >( value.GetUint() );
  }
  if( value.IsNumber() )
  {
    return static_cast< int >( value.GetDouble() );
  }
  return 0;
}

double
json_double( const json_value& value )
{
  return value.IsNumber() ? value.GetDouble() : 0.0;
}

std::string
json_id( const json_value& value )
{
  if( value.IsString() )
  {
    return std::string( value.GetString(), value.GetStringLength() );
  }
  if( value.IsInt() )
  {
    return std::to_string( value.GetInt() );
  }
  if( value.IsInt64() )
  {
    return std::to_string( value.GetInt64() );
  }
  if( value.IsUint() )
  {
    return std::to_string( value.GetUint() );
  }
  if( value.IsUint64() )
  {
    return std::to_string( value.GetUint64() );
  }
  if( value.IsNumber() )
  {
    std::ostringstream oss;
    oss << value.GetDouble();
    return oss.str();
  }
  return "";
}

void
set_int( json_value& object, const char* name, int number, json_alloc& alloc )
{
  auto itr = object.FindMember( name );

  if( itr != object.MemberEnd() )
  {
    itr->value.SetInt( number );
  }
  else
  {
    object.AddMember( rapidjson::StringRef( name ), number, alloc );
  }
}

void
set_string( json_value& object, const char* name, const std::string& text, json_alloc& alloc )
{
  json_value value( text.c_str(), alloc );
  auto itr = object.FindMember( name );

  if( itr != object.MemberEnd() )
  {
    itr->value = value;
  }
  else
  {
    object.AddMember( rapidjson::StringRef( name ), value, alloc );
  }
}

void
shift_int( json_value& object, const char* name, int delta )
{
  json_value* value = find( object, name );

  if( value && value->IsNumber() )
  {
    value->SetInt( json_int( *value ) + delta );
  }
}

json_value
single_confidence_pair( const std::string& name, json_alloc& alloc )
{
  json_value pair( rapidjson::kArrayType );
  pair.PushBack( json_value( name.c_str(), alloc ), alloc );
  pair.PushBack( json_value( 1.0 ), alloc );

  json_value pairs( rapidjson::kArrayType );
  pairs.PushBack( pair, alloc );
  return pairs;
}

// =======================================================================================
// Normalized view of one annotation, shared by every report and filter

struct entry
{
  std::string track_key;
  std::string track_id;
  std::string image_key;
  int frame = -1;
  double x1 = 0.0;
  double y1 = 0.0;
  double x2 = 0.0;
  double y2 = 0.0;
  double confidence = 1.0;
  std::string top_type;
  double top_score = -1.0;
  size_t index = 0;
};

using drop_set = std::set< std::pair< std::string, size_t > >;
using name_map = std::map< std::string, std::string >;

// =======================================================================================
class format_adapter
{
public:
  format_adapter( const std::string& file ) : m_file( file ) {}
  virtual ~format_adapter() {}

  virtual std::vector< entry > extract( json_doc& doc ) = 0;
  virtual void remove( json_doc& doc, const drop_set& dropped ) = 0;
  virtual void require_frames( const json_doc& doc ) = 0;
  virtual void shift_frames( json_doc& doc, int delta ) = 0;
  virtual void filter_frame_range( json_doc& doc, int lower, int upper ) = 0;
  virtual void renumber_tracks( json_doc& doc, const name_map& id_map ) = 0;
  virtual void replace_types( json_doc& doc, const name_map& replacements ) = 0;
  virtual double fps( const json_doc& doc ) = 0;
  virtual std::vector< std::string > validate( const json_doc& doc ) = 0;

protected:
  std::string m_file;
};

// =======================================================================================
// DIVE format

int
dive_feature_frame( const json_value& feature )
{
  const json_value* frame = find( feature, "frame" );
  return ( frame && frame->IsNumber() ) ? json_int( *frame ) : 0;
}

std::string
dive_track_id( const json_value& track, const std::string& key )
{
  const json_value* id = find( track, "id" );
  return ( id && id->IsNumber() ) ? json_id( *id ) : key;
}

bool
dive_top_pair( const json_value& track, std::string& name, double& score )
{
  const json_value* pairs = find( track, "confidencePairs" );

  if( !pairs || !pairs->IsArray() )
  {
    return false;
  }

  bool found = false;

  for( const auto& pair : pairs->GetArray() )
  {
    if( pair.IsArray() && pair.Size() >= 2 && pair[0].IsString() && pair[1].IsNumber() )
    {
      const double value = pair[1].GetDouble();

      if( !found || value > score )
      {
        name = pair[0].GetString();
        score = value;
        found = true;
      }
    }
  }

  return found;
}

void
dive_sort_features( json_value& features, json_alloc& alloc )
{
  std::vector< std::pair< int, size_t > > order;

  for( json_size i = 0; i < features.Size(); ++i )
  {
    order.push_back( std::make_pair( dive_feature_frame( features[i] ), static_cast< size_t >( i ) ) );
  }

  if( std::is_sorted( order.begin(), order.end() ) )
  {
    return;
  }

  std::stable_sort( order.begin(), order.end(),
    []( const std::pair< int, size_t >& a, const std::pair< int, size_t >& b )
    {
      return a.first < b.first;
    } );

  json_value sorted( rapidjson::kArrayType );
  sorted.Reserve( features.Size(), alloc );

  for( const auto& item : order )
  {
    sorted.PushBack( features[ static_cast< json_size >( item.second ) ], alloc );
  }

  features = sorted;
}

class dive_adapter : public format_adapter
{
public:
  dive_adapter( const std::string& file ) : format_adapter( file ) {}

  // -------------------------------------------------------------------------------------
  std::vector< entry > extract( json_doc& doc ) override
  {
    std::vector< entry > entries;
    json_value* tracks = find( doc, "tracks" );

    if( !tracks || !tracks->IsObject() )
    {
      throw std::runtime_error( m_file + ": no 'tracks' object" );
    }

    for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
    {
      const std::string key = itr->name.GetString();
      json_value& track = itr->value;

      if( !track.IsObject() )
      {
        continue;
      }

      const std::string id = dive_track_id( track, key );

      std::string type;
      double score = -1.0;
      const bool typed = dive_top_pair( track, type, score );

      json_value* features = find( track, "features" );

      if( !features || !features->IsArray() )
      {
        continue;
      }

      for( json_size i = 0; i < features->Size(); ++i )
      {
        const json_value& feature = ( *features )[i];

        entry item;
        item.track_key = key;
        item.track_id = id;
        item.index = static_cast< size_t >( i );
        item.frame = dive_feature_frame( feature );
        item.image_key = std::to_string( item.frame );
        item.confidence = typed ? score : 1.0;
        item.top_type = typed ? type : "";
        item.top_score = typed ? score : -1.0;

        const json_value* bounds = find( feature, "bounds" );

        if( !bounds || !bounds->IsArray() || bounds->Size() < 4 )
        {
          throw std::runtime_error( m_file + ": track " + id + " frame " +
            std::to_string( item.frame ) + " has malformed bounds" );
        }

        item.x1 = json_double( ( *bounds )[0] );
        item.y1 = json_double( ( *bounds )[1] );
        item.x2 = json_double( ( *bounds )[2] );
        item.y2 = json_double( ( *bounds )[3] );

        entries.push_back( item );
      }
    }

    return entries;
  }

  // -------------------------------------------------------------------------------------
  void remove( json_doc& doc, const drop_set& dropped ) override
  {
    json_value* tracks = find( doc, "tracks" );

    if( tracks && tracks->IsObject() )
    {
      for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
      {
        const std::string key = itr->name.GetString();
        json_value* features = find( itr->value, "features" );

        if( !features || !features->IsArray() )
        {
          continue;
        }

        for( int i = static_cast< int >( features->Size() ) - 1; i >= 0; --i )
        {
          if( dropped.count( std::make_pair( key, static_cast< size_t >( i ) ) ) )
          {
            features->Erase( features->Begin() + i );
          }
        }
      }
    }

    fix_extents( doc );
  }

  // -------------------------------------------------------------------------------------
  void require_frames( const json_doc& doc ) override
  {
    const json_value* tracks = find( doc, "tracks" );

    if( !tracks || !tracks->IsObject() )
    {
      return;
    }

    for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
    {
      const json_value* features = find( itr->value, "features" );

      if( !features || !features->IsArray() )
      {
        continue;
      }

      for( json_size i = 0; i < features->Size(); ++i )
      {
        const json_value* frame = find( ( *features )[i], "frame" );

        if( !frame || !frame->IsNumber() )
        {
          throw std::runtime_error( m_file + ": track " +
            dive_track_id( itr->value, itr->name.GetString() ) + " feature " +
            std::to_string( i ) + " has no frame" );
        }
      }
    }
  }

  // -------------------------------------------------------------------------------------
  void shift_frames( json_doc& doc, int delta ) override
  {
    json_value* tracks = find( doc, "tracks" );

    if( tracks && tracks->IsObject() )
    {
      for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
      {
        shift_int( itr->value, "begin", delta );
        shift_int( itr->value, "end", delta );

        json_value* features = find( itr->value, "features" );

        if( features && features->IsArray() )
        {
          for( auto& feature : features->GetArray() )
          {
            shift_int( feature, "frame", delta );
          }
        }
      }
    }

    json_value* groups = find( doc, "groups" );

    if( !groups || !groups->IsObject() )
    {
      return;
    }

    for( auto itr = groups->MemberBegin(); itr != groups->MemberEnd(); ++itr )
    {
      shift_int( itr->value, "begin", delta );
      shift_int( itr->value, "end", delta );

      json_value* members = find( itr->value, "members" );

      if( !members || !members->IsObject() )
      {
        continue;
      }

      for( auto member = members->MemberBegin(); member != members->MemberEnd(); ++member )
      {
        json_value* ranges = find( member->value, "ranges" );

        if( !ranges || !ranges->IsArray() )
        {
          continue;
        }

        for( auto& range : ranges->GetArray() )
        {
          if( !range.IsArray() )
          {
            continue;
          }

          for( auto& bound : range.GetArray() )
          {
            if( bound.IsNumber() )
            {
              bound.SetInt( json_int( bound ) + delta );
            }
          }
        }
      }
    }
  }

  // -------------------------------------------------------------------------------------
  void filter_frame_range( json_doc& doc, int lower, int upper ) override
  {
    json_value* tracks = find( doc, "tracks" );

    if( tracks && tracks->IsObject() )
    {
      for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
      {
        json_value* features = find( itr->value, "features" );

        if( !features || !features->IsArray() )
        {
          continue;
        }

        for( int i = static_cast< int >( features->Size() ) - 1; i >= 0; --i )
        {
          const int frame =
            dive_feature_frame( ( *features )[ static_cast< json_size >( i ) ] );

          if( frame < lower || frame > upper )
          {
            features->Erase( features->Begin() + i );
          }
        }
      }
    }

    fix_extents( doc );

    if( lower != 0 )
    {
      shift_frames( doc, -lower );
    }
  }

  // -------------------------------------------------------------------------------------
  void renumber_tracks( json_doc& doc, const name_map& id_map ) override
  {
    json_alloc& alloc = doc.GetAllocator();
    json_value* tracks = find( doc, "tracks" );

    if( !tracks || !tracks->IsObject() )
    {
      return;
    }

    name_map applied;
    json_value rebuilt( rapidjson::kObjectType );

    for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
    {
      const std::string key = itr->name.GetString();
      const std::string id = dive_track_id( itr->value, key );

      auto match = id_map.find( id );
      std::string assigned = key;

      if( match != id_map.end() )
      {
        assigned = match->second;
        applied[ key ] = assigned;
        applied[ id ] = assigned;

        if( itr->value.IsObject() )
        {
          set_int( itr->value, "id", std::atoi( assigned.c_str() ), alloc );
        }
      }

      json_value name( assigned.c_str(), alloc );
      rebuilt.AddMember( name, itr->value, alloc );
    }

    *tracks = rebuilt;

    json_value* groups = find( doc, "groups" );

    if( !groups || !groups->IsObject() )
    {
      return;
    }

    for( auto itr = groups->MemberBegin(); itr != groups->MemberEnd(); ++itr )
    {
      json_value* members = find( itr->value, "members" );

      if( !members || !members->IsObject() )
      {
        continue;
      }

      json_value remapped( rapidjson::kObjectType );

      for( auto member = members->MemberBegin(); member != members->MemberEnd(); ++member )
      {
        const std::string key = member->name.GetString();
        auto match = applied.find( key );

        json_value name( ( match != applied.end() ? match->second : key ).c_str(), alloc );
        remapped.AddMember( name, member->value, alloc );
      }

      *members = remapped;
    }
  }

  // -------------------------------------------------------------------------------------
  void replace_types( json_doc& doc, const name_map& replacements ) override
  {
    json_alloc& alloc = doc.GetAllocator();
    json_value* tracks = find( doc, "tracks" );

    if( !tracks || !tracks->IsObject() )
    {
      return;
    }

    for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
    {
      std::string type;
      double score = -1.0;

      if( !dive_top_pair( itr->value, type, score ) )
      {
        continue;
      }

      auto match = replacements.find( type );
      const std::string assigned = ( match != replacements.end() ) ? match->second : type;

      json_value pairs = single_confidence_pair( assigned, alloc );
      json_value* existing = find( itr->value, "confidencePairs" );

      if( existing )
      {
        *existing = pairs;
      }
      else
      {
        itr->value.AddMember( rapidjson::StringRef( "confidencePairs" ), pairs, alloc );
      }
    }
  }

  // -------------------------------------------------------------------------------------
  double fps( const json_doc& doc ) override
  {
    const json_value* rate = find( doc, "fps" );

    if( rate && rate->IsNumber() && rate->GetDouble() > 0.0 )
    {
      return rate->GetDouble();
    }

    return -1.0;
  }

  // -------------------------------------------------------------------------------------
  std::vector< std::string > validate( const json_doc& doc ) override
  {
    std::vector< std::string > messages;
    const json_value* tracks = find( doc, "tracks" );

    if( !tracks || !tracks->IsObject() )
    {
      messages.push_back( m_file + ": no 'tracks' object" );
      return messages;
    }

    std::set< std::string > known;

    for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); ++itr )
    {
      const std::string key = itr->name.GetString();
      const json_value& track = itr->value;

      if( !track.IsObject() )
      {
        messages.push_back( m_file + ": track " + key + ": value is not an object" );
        continue;
      }

      const std::string id = dive_track_id( track, key );
      const std::string prefix = m_file + ": track " + id + ": ";

      known.insert( id );
      known.insert( key );

      if( key != id )
      {
        messages.push_back( prefix + "member key '" + key + "' does not match the track id" );
      }

      const json_value* pairs = find( track, "confidencePairs" );

      if( !pairs || !pairs->IsArray() || pairs->Size() == 0 )
      {
        messages.push_back( prefix + "no confidencePairs" );
      }
      else
      {
        for( const auto& pair : pairs->GetArray() )
        {
          if( !pair.IsArray() || pair.Size() < 2 ||
              !pair[0].IsString() || !pair[1].IsNumber() )
          {
            messages.push_back( prefix + "confidencePairs entry is not [name, score]" );
            break;
          }
        }
      }

      const json_value* features = find( track, "features" );

      if( !features || !features->IsArray() || features->Size() == 0 )
      {
        messages.push_back( prefix + "no features" );
        continue;
      }

      int previous = std::numeric_limits< int >::min();
      bool sorted = true;

      for( json_size i = 0; i < features->Size(); ++i )
      {
        const json_value& feature = ( *features )[i];
        const json_value* frame = find( feature, "frame" );

        if( !frame || !frame->IsNumber() )
        {
          messages.push_back( prefix + "feature " + std::to_string( i ) + " has no frame" );
          continue;
        }

        const int value = json_int( *frame );

        if( value < previous )
        {
          sorted = false;
        }
        previous = value;

        const json_value* bounds = find( feature, "bounds" );

        if( !bounds || !bounds->IsArray() || bounds->Size() < 4 )
        {
          messages.push_back( prefix + "feature at frame " + std::to_string( value ) +
            " has malformed bounds" );
        }
      }

      if( !sorted )
      {
        messages.push_back( prefix + "features are not ordered by frame" );
      }

      const int begin = dive_feature_frame( ( *features )[0] );
      const int end = dive_feature_frame( ( *features )[ features->Size() - 1 ] );

      const json_value* begin_value = find( track, "begin" );
      const json_value* end_value = find( track, "end" );

      if( !begin_value || !begin_value->IsNumber() || json_int( *begin_value ) != begin )
      {
        messages.push_back( prefix + "begin does not match the first feature frame (" +
          std::to_string( begin ) + ")" );
      }

      if( !end_value || !end_value->IsNumber() || json_int( *end_value ) != end )
      {
        messages.push_back( prefix + "end does not match the last feature frame (" +
          std::to_string( end ) + ")" );
      }
    }

    const json_value* groups = find( doc, "groups" );

    if( groups && groups->IsObject() )
    {
      for( auto itr = groups->MemberBegin(); itr != groups->MemberEnd(); ++itr )
      {
        const json_value* members = find( itr->value, "members" );

        if( !members || !members->IsObject() )
        {
          continue;
        }

        for( auto member = members->MemberBegin(); member != members->MemberEnd(); ++member )
        {
          if( !known.count( member->name.GetString() ) )
          {
            messages.push_back( m_file + ": group " + itr->name.GetString() +
              ": member references unknown track " + member->name.GetString() );
          }
        }
      }
    }

    return messages;
  }

private:
  // -------------------------------------------------------------------------------------
  // DIVE readers reject begin/end that disagree with the features, so every
  // structural edit ends here
  void fix_extents( json_doc& doc )
  {
    json_alloc& alloc = doc.GetAllocator();
    json_value* tracks = find( doc, "tracks" );

    if( !tracks || !tracks->IsObject() )
    {
      return;
    }

    std::set< std::string > removed;

    for( auto itr = tracks->MemberBegin(); itr != tracks->MemberEnd(); )
    {
      json_value* features = find( itr->value, "features" );

      if( !features || !features->IsArray() || features->Size() == 0 )
      {
        removed.insert( itr->name.GetString() );
        removed.insert( dive_track_id( itr->value, itr->name.GetString() ) );
        itr = tracks->EraseMember( itr );
        continue;
      }

      dive_sort_features( *features, alloc );
      set_int( itr->value, "begin", dive_feature_frame( ( *features )[0] ), alloc );
      set_int( itr->value, "end",
        dive_feature_frame( ( *features )[ features->Size() - 1 ] ), alloc );
      ++itr;
    }

    if( removed.empty() )
    {
      return;
    }

    json_value* groups = find( doc, "groups" );

    if( !groups || !groups->IsObject() )
    {
      return;
    }

    for( auto itr = groups->MemberBegin(); itr != groups->MemberEnd(); )
    {
      json_value* members = find( itr->value, "members" );

      if( !members || !members->IsObject() )
      {
        ++itr;
        continue;
      }

      for( auto member = members->MemberBegin(); member != members->MemberEnd(); )
      {
        if( removed.count( member->name.GetString() ) )
        {
          member = members->EraseMember( member );
        }
        else
        {
          ++member;
        }
      }

      if( members->MemberCount() == 0 )
      {
        itr = groups->EraseMember( itr );
      }
      else
      {
        ++itr;
      }
    }
  }
};

// =======================================================================================
// COCO format

class coco_adapter : public format_adapter
{
public:
  coco_adapter( const std::string& file ) : format_adapter( file ) {}

  // -------------------------------------------------------------------------------------
  std::vector< entry > extract( json_doc& doc ) override
  {
    build_index( doc );

    std::vector< entry > entries;
    json_value* annotations = find( doc, "annotations" );

    if( !annotations || !annotations->IsArray() )
    {
      throw std::runtime_error( m_file + ": no 'annotations' array" );
    }

    for( json_size i = 0; i < annotations->Size(); ++i )
    {
      const json_value& annotation = ( *annotations )[i];

      const json_value* id = find( annotation, "id" );
      const std::string annotation_id = id ? json_id( *id ) : std::to_string( i );

      entry item;
      item.index = static_cast< size_t >( i );

      const json_value* track = find( annotation, "track_id" );
      item.track_id = track ? json_id( *track ) : annotation_id;

      const json_value* image_id = find( annotation, "image_id" );
      const std::string image_key = image_id ? json_id( *image_id ) : "";
      auto image = m_images.find( image_key );

      if( image == m_images.end() )
      {
        throw std::runtime_error( m_file + ": annotation " + annotation_id +
          " references unknown image " + image_key );
      }

      item.image_key = image->second.file_name.empty()
        ? ( "image " + image_key ) : image->second.file_name;
      item.frame = image->second.has_frame ? image->second.frame : -1;

      const json_value* bbox = find( annotation, "bbox" );

      if( !bbox || !bbox->IsArray() || bbox->Size() < 4 )
      {
        throw std::runtime_error( m_file + ": annotation " + annotation_id +
          " has malformed bbox" );
      }

      item.x1 = json_double( ( *bbox )[0] );
      item.y1 = json_double( ( *bbox )[1] );
      item.x2 = item.x1 + json_double( ( *bbox )[2] );
      item.y2 = item.y1 + json_double( ( *bbox )[3] );

      const json_value* score = find( annotation, "score" );
      item.confidence = ( score && score->IsNumber() ) ? score->GetDouble() : 1.0;

      const json_value* category = find( annotation, "category_id" );
      const std::string category_id = category ? json_id( *category ) : "";
      auto named = m_categories.find( category_id );

      if( named == m_categories.end() )
      {
        throw std::runtime_error( m_file + ": annotation " + annotation_id +
          " references unknown category " + category_id );
      }

      item.top_type = named->second;
      item.top_score = item.confidence;

      double best = 0.0;

      if( highest( find( annotation, "confidence_pairs" ), true, best ) ||
          highest( find( annotation, "prob" ), false, best ) )
      {
        item.top_score = best;
      }

      entries.push_back( item );
    }

    return entries;
  }

  // -------------------------------------------------------------------------------------
  void remove( json_doc& doc, const drop_set& dropped ) override
  {
    json_value* annotations = find( doc, "annotations" );

    if( !annotations || !annotations->IsArray() )
    {
      return;
    }

    for( int i = static_cast< int >( annotations->Size() ) - 1; i >= 0; --i )
    {
      if( dropped.count( std::make_pair( std::string(), static_cast< size_t >( i ) ) ) )
      {
        annotations->Erase( annotations->Begin() + i );
      }
    }
  }

  // -------------------------------------------------------------------------------------
  void require_frames( const json_doc& doc ) override
  {
    const json_value* images = find( doc, "images" );

    if( !images || !images->IsArray() )
    {
      return;
    }

    for( const auto& image : images->GetArray() )
    {
      const json_value* frame = find( image, "frame_index" );

      if( !frame || !frame->IsNumber() )
      {
        const json_value* id = find( image, "id" );
        throw std::runtime_error( m_file + ": image " +
          ( id ? json_id( *id ) : std::string( "?" ) ) + " has no frame_index" );
      }
    }
  }

  // -------------------------------------------------------------------------------------
  void shift_frames( json_doc& doc, int delta ) override
  {
    json_value* images = find( doc, "images" );

    if( !images || !images->IsArray() )
    {
      return;
    }

    for( auto& image : images->GetArray() )
    {
      shift_int( image, "frame_index", delta );
    }
  }

  // -------------------------------------------------------------------------------------
  void filter_frame_range( json_doc& doc, int lower, int upper ) override
  {
    build_index( doc );

    std::set< std::string > dropped;

    for( const auto& image : m_images )
    {
      if( !image.second.has_frame ||
          image.second.frame < lower || image.second.frame > upper )
      {
        dropped.insert( image.first );
      }
    }

    json_value* annotations = find( doc, "annotations" );

    if( annotations && annotations->IsArray() )
    {
      for( int i = static_cast< int >( annotations->Size() ) - 1; i >= 0; --i )
      {
        const json_value* image_id =
          find( ( *annotations )[ static_cast< json_size >( i ) ], "image_id" );

        if( image_id && dropped.count( json_id( *image_id ) ) )
        {
          annotations->Erase( annotations->Begin() + i );
        }
      }
    }

    json_value* images = find( doc, "images" );

    if( images && images->IsArray() )
    {
      for( int i = static_cast< int >( images->Size() ) - 1; i >= 0; --i )
      {
        const json_value* id = find( ( *images )[ static_cast< json_size >( i ) ], "id" );

        if( id && dropped.count( json_id( *id ) ) )
        {
          images->Erase( images->Begin() + i );
        }
      }
    }

    if( lower != 0 )
    {
      shift_frames( doc, -lower );
    }
  }

  // -------------------------------------------------------------------------------------
  void renumber_tracks( json_doc& doc, const name_map& id_map ) override
  {
    json_alloc& alloc = doc.GetAllocator();
    json_value* annotations = find( doc, "annotations" );

    if( !annotations || !annotations->IsArray() )
    {
      return;
    }

    std::vector< std::string > assigned;
    std::set< std::string > seen;

    for( auto& annotation : annotations->GetArray() )
    {
      const json_value* id = find( annotation, "id" );
      const json_value* track = find( annotation, "track_id" );

      std::string current;

      if( track )
      {
        current = json_id( *track );
      }
      else if( id )
      {
        current = json_id( *id );
      }

      auto match = id_map.find( current );

      if( match == id_map.end() )
      {
        continue;
      }

      if( seen.insert( match->second ).second )
      {
        assigned.push_back( match->second );
      }

      set_int( annotation, "track_id", std::atoi( match->second.c_str() ), alloc );
    }

    json_value* tracks = find( doc, "tracks" );

    if( !tracks || !tracks->IsArray() )
    {
      return;
    }

    std::set< std::string > present;

    for( auto& track : tracks->GetArray() )
    {
      json_value* id = find( track, "id" );

      if( !id )
      {
        continue;
      }

      const std::string previous = json_id( *id );
      auto match = id_map.find( previous );

      if( match != id_map.end() )
      {
        id->SetInt( std::atoi( match->second.c_str() ) );

        const json_value* name = find( track, "name" );

        if( name && name->IsString() && previous == name->GetString() )
        {
          set_string( track, "name", match->second, alloc );
        }
      }

      present.insert( json_id( *id ) );
    }

    for( const auto& id : assigned )
    {
      if( present.count( id ) )
      {
        continue;
      }

      json_value record( rapidjson::kObjectType );
      record.AddMember( rapidjson::StringRef( "id" ), std::atoi( id.c_str() ), alloc );
      record.AddMember( rapidjson::StringRef( "name" ), json_value( id.c_str(), alloc ), alloc );
      tracks->PushBack( record, alloc );
    }
  }

  // -------------------------------------------------------------------------------------
  void replace_types( json_doc& doc, const name_map& replacements ) override
  {
    json_alloc& alloc = doc.GetAllocator();
    json_value* categories = find( doc, "categories" );

    if( !categories || !categories->IsArray() )
    {
      return;
    }

    std::map< std::string, int > merged;
    std::map< std::string, int > remap;
    std::vector< std::string > ordered;
    json_value rebuilt( rapidjson::kArrayType );

    for( auto& category : categories->GetArray() )
    {
      const json_value* id = find( category, "id" );
      const json_value* name = find( category, "name" );

      if( !id || !name || !name->IsString() )
      {
        continue;
      }

      const std::string category_id = json_id( *id );
      const std::string original = name->GetString();

      auto match = replacements.find( original );
      const std::string assigned = ( match != replacements.end() ) ? match->second : original;

      auto known = merged.find( assigned );

      if( known != merged.end() )
      {
        remap[ category_id ] = known->second;
        continue;
      }

      const int new_id = static_cast< int >( merged.size() ) + 1;
      merged[ assigned ] = new_id;
      remap[ category_id ] = new_id;
      ordered.push_back( assigned );

      json_value kept;
      kept = category;
      set_int( kept, "id", new_id, alloc );
      set_string( kept, "name", assigned, alloc );
      rebuilt.PushBack( kept, alloc );
    }

    *categories = rebuilt;

    json_value* annotations = find( doc, "annotations" );

    if( !annotations || !annotations->IsArray() )
    {
      return;
    }

    for( auto& annotation : annotations->GetArray() )
    {
      json_value* category = find( annotation, "category_id" );

      if( !category )
      {
        continue;
      }

      auto match = remap.find( json_id( *category ) );

      if( match == remap.end() )
      {
        continue;
      }

      const int new_id = match->second;
      category->SetInt( new_id );

      json_value* pairs = find( annotation, "confidence_pairs" );

      if( pairs )
      {
        *pairs = single_confidence_pair( ordered[ new_id - 1 ], alloc );
      }

      json_value* prob = find( annotation, "prob" );

      if( prob )
      {
        json_value vector( rapidjson::kArrayType );

        for( size_t i = 0; i < ordered.size(); ++i )
        {
          vector.PushBack(
            json_value( i == static_cast< size_t >( new_id - 1 ) ? 1.0 : 0.0 ), alloc );
        }

        *prob = vector;
      }
    }
  }

  // -------------------------------------------------------------------------------------
  double fps( const json_doc& doc ) override
  {
    const json_value* videos = find( doc, "videos" );

    if( !videos || !videos->IsArray() )
    {
      return -1.0;
    }

    for( const auto& video : videos->GetArray() )
    {
      const json_value* rate = find( video, "annotation_fps" );

      if( rate && rate->IsNumber() && rate->GetDouble() > 0.0 )
      {
        return rate->GetDouble();
      }
    }

    return -1.0;
  }

  // -------------------------------------------------------------------------------------
  std::vector< std::string > validate( const json_doc& doc ) override
  {
    std::vector< std::string > messages;

    const json_value* images = find( doc, "images" );
    const json_value* annotations = find( doc, "annotations" );
    const json_value* categories = find( doc, "categories" );

    if( !images || !images->IsArray() )
    {
      messages.push_back( m_file + ": no 'images' array" );
    }
    if( !annotations || !annotations->IsArray() )
    {
      messages.push_back( m_file + ": no 'annotations' array" );
    }
    if( !categories || !categories->IsArray() )
    {
      messages.push_back( m_file + ": no 'categories' array" );
    }

    if( !messages.empty() )
    {
      return messages;
    }

    std::set< std::string > image_ids;
    int with_frame = 0;
    int without_frame = 0;

    for( const auto& image : images->GetArray() )
    {
      const json_value* id = find( image, "id" );

      if( !id )
      {
        messages.push_back( m_file + ": an image entry has no id" );
        continue;
      }

      const std::string key = json_id( *id );
      const std::string prefix = m_file + ": image " + key + ": ";

      if( !image_ids.insert( key ).second )
      {
        messages.push_back( prefix + "duplicate id" );
      }

      const json_value* name = find( image, "file_name" );

      if( !name || !name->IsString() || name->GetStringLength() == 0 )
      {
        messages.push_back( prefix + "no file_name" );
      }

      const json_value* frame = find( image, "frame_index" );

      if( frame && frame->IsNumber() )
      {
        ++with_frame;
      }
      else
      {
        ++without_frame;
      }
    }

    if( with_frame > 0 && without_frame > 0 )
    {
      messages.push_back( m_file + ": some images carry frame_index and some do not" );
    }

    std::set< std::string > category_ids;

    for( const auto& category : categories->GetArray() )
    {
      const json_value* id = find( category, "id" );

      if( !id )
      {
        messages.push_back( m_file + ": a category entry has no id" );
        continue;
      }

      const std::string key = json_id( *id );

      if( !category_ids.insert( key ).second )
      {
        messages.push_back( m_file + ": category " + key + ": duplicate id" );
      }

      const json_value* name = find( category, "name" );

      if( !name || !name->IsString() )
      {
        messages.push_back( m_file + ": category " + key + ": no name" );
      }
    }

    std::set< std::string > annotation_ids;

    for( const auto& annotation : annotations->GetArray() )
    {
      const json_value* id = find( annotation, "id" );

      if( !id )
      {
        messages.push_back( m_file + ": an annotation entry has no id" );
        continue;
      }

      const std::string key = json_id( *id );
      const std::string prefix = m_file + ": annotation " + key + ": ";

      if( !annotation_ids.insert( key ).second )
      {
        messages.push_back( prefix + "duplicate id" );
      }

      const json_value* image_id = find( annotation, "image_id" );

      if( !image_id || !image_ids.count( json_id( *image_id ) ) )
      {
        messages.push_back( prefix + "references unknown image " +
          ( image_id ? json_id( *image_id ) : std::string() ) );
      }

      const json_value* category_id = find( annotation, "category_id" );

      if( !category_id || !category_ids.count( json_id( *category_id ) ) )
      {
        messages.push_back( prefix + "references unknown category " +
          ( category_id ? json_id( *category_id ) : std::string() ) );
      }

      const json_value* bbox = find( annotation, "bbox" );

      if( !bbox || !bbox->IsArray() || bbox->Size() < 4 )
      {
        messages.push_back( prefix + "bbox is not four numbers" );
      }
    }

    return messages;
  }

private:
  struct image_info
  {
    std::string file_name;
    int frame = -1;
    bool has_frame = false;
  };

  // -------------------------------------------------------------------------------------
  static bool highest( const json_value* values, bool paired, double& best )
  {
    if( !values || !values->IsArray() || values->Size() == 0 )
    {
      return false;
    }

    bool found = false;

    for( const auto& value : values->GetArray() )
    {
      const json_value* score = nullptr;

      if( paired )
      {
        if( value.IsArray() && value.Size() >= 2 && value[1].IsNumber() )
        {
          score = &value[1];
        }
      }
      else if( value.IsNumber() )
      {
        score = &value;
      }

      if( score && ( !found || score->GetDouble() > best ) )
      {
        best = score->GetDouble();
        found = true;
      }
    }

    return found;
  }

  // -------------------------------------------------------------------------------------
  void build_index( const json_doc& doc )
  {
    m_images.clear();
    m_categories.clear();

    const json_value* images = find( doc, "images" );

    if( images && images->IsArray() )
    {
      for( const auto& image : images->GetArray() )
      {
        const json_value* id = find( image, "id" );

        if( !id )
        {
          continue;
        }

        image_info info;
        const json_value* name = find( image, "file_name" );

        if( name && name->IsString() )
        {
          info.file_name = name->GetString();
        }

        const json_value* frame = find( image, "frame_index" );

        if( frame && frame->IsNumber() )
        {
          info.frame = json_int( *frame );
          info.has_frame = true;
        }

        m_images[ json_id( *id ) ] = info;
      }
    }

    const json_value* categories = find( doc, "categories" );

    if( categories && categories->IsArray() )
    {
      for( const auto& category : categories->GetArray() )
      {
        const json_value* id = find( category, "id" );
        const json_value* name = find( category, "name" );

        if( id && name && name->IsString() )
        {
          m_categories[ json_id( *id ) ] = name->GetString();
        }
      }
    }
  }

  std::map< std::string, image_info > m_images;
  std::map< std::string, std::string > m_categories;
};

// =======================================================================================
// Document input and output

json_doc
load_document( const std::string& filename )
{
  FILE* handle = std::fopen( filename.c_str(), "rb" );

  if( !handle )
  {
    throw std::runtime_error( filename + ": could not open file" );
  }

  std::vector< char > buffer( 65536 );
  rapidjson::FileReadStream stream( handle, buffer.data(),
    static_cast< size_t >( buffer.size() ) );

  json_doc doc;
  doc.ParseStream< rapidjson::kParseFullPrecisionFlag >( stream );
  std::fclose( handle );

  if( doc.HasParseError() )
  {
    throw std::runtime_error( filename + ": " +
      rapidjson::GetParseError_En( doc.GetParseError() ) + " at offset " +
      std::to_string( doc.GetErrorOffset() ) );
  }

  if( !doc.IsObject() )
  {
    throw std::runtime_error( filename + ": root value is not a JSON object" );
  }

  return doc;
}

void
save_document( const std::string& filename, const json_doc& doc )
{
  viame::atomic_output( filename, [&]( std::ostream& fout )
  {
    rapidjson::OStreamWrapper wrapper( fout );
    rapidjson::PrettyWriter< rapidjson::OStreamWrapper > writer( wrapper );
    writer.SetIndent( ' ', 2 );
    if( !doc.Accept( writer ) )
    {
      throw std::runtime_error( filename + ": could not serialize JSON" );
    }
    fout << "\n";
  } );
}

// =======================================================================================
std::string
detect_format( const json_doc& doc )
{
  const json_value* images = find( doc, "images" );
  const json_value* annotations = find( doc, "annotations" );
  const json_value* categories = find( doc, "categories" );

  if( images && images->IsArray() &&
      annotations && annotations->IsArray() &&
      categories && categories->IsArray() )
  {
    return "coco";
  }

  const json_value* tracks = find( doc, "tracks" );

  if( tracks && tracks->IsObject() )
  {
    return "dive";
  }

  const json_value* version = find( doc, "version" );

  if( version && version->IsNumber() && json_int( *version ) == 2 )
  {
    return "dive";
  }

  for( auto itr = doc.MemberBegin(); itr != doc.MemberEnd(); ++itr )
  {
    if( itr->value.IsObject() && itr->value.HasMember( "trackId" ) )
    {
      return "dive_v1";
    }
  }

  return "";
}

// =======================================================================================
void
upgrade_dive_v1( json_doc& doc )
{
  json_alloc& alloc = doc.GetAllocator();
  json_value tracks( rapidjson::kObjectType );

  for( auto itr = doc.MemberBegin(); itr != doc.MemberEnd(); ++itr )
  {
    json_value& track = itr->value;

    if( track.IsObject() )
    {
      auto legacy = track.FindMember( "trackId" );

      if( legacy != track.MemberEnd() )
      {
        json_value id;
        id = legacy->value;
        track.EraseMember( legacy );

        json_value* existing = find( track, "id" );

        if( existing )
        {
          *existing = id;
        }
        else
        {
          track.AddMember( rapidjson::StringRef( "id" ), id, alloc );
        }
      }
    }

    json_value name( itr->name, alloc );
    tracks.AddMember( name, track, alloc );
  }

  doc.RemoveAllMembers();
  doc.AddMember( rapidjson::StringRef( "version" ), 2, alloc );
  doc.AddMember( rapidjson::StringRef( "tracks" ), tracks, alloc );
  doc.AddMember( rapidjson::StringRef( "groups" ),
    json_value( rapidjson::kObjectType ), alloc );
}

} // anonymous namespace

// =======================================================================================
void
json_applet
::add_command_options()
{
  m_cmd_options->add_options()
    ( "h,help", "Display usage information",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "i,input", "Input JSON file, directory, or file name pattern to process. "
      "Wildcards apply to the file name only, not to directories",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "decrease-fid", "Decrease frame IDs in files by 1",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "increase-fid", "Increase frame IDs in files by 1",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "assign-uid", "Assign unique detection IDs to all entries in volume",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "filter-single", "Filter single state tracks",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "print-types", "Print unique list of target types",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "caps-only", "Only print types with capitalized letters in them",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "track-count", "Print total number of tracks",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "counts-per-frame", "Print total number of detections per frame",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "average-box-size", "Print average box size per type",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "conf-threshold", "Confidence threshold",
      ::cxxopts::value< double >()->default_value( "-1.0" ), "value" )
    ( "type-threshold", "Type confidence threshold",
      ::cxxopts::value< double >()->default_value( "-1.0" ), "value" )
    ( "print-filtered", "Print out tracks that were filtered out",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "print-single", "Print out video sequences only containing single states",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "lower-fid", "Lower FID if adjusting FIDs to be within some range",
      ::cxxopts::value< int >()->default_value( "0" ), "value" )
    ( "upper-fid", "Upper FID if adjusting FIDs to be within some range",
      ::cxxopts::value< int >()->default_value( "0" ), "value" )
    ( "replace-file", "If set, replace all types in this file given their synonyms",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "print-fps", "Print FPS in input files",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ( "comp-file", "If set, generate a comparison file contrasting types in all inputs",
      ::cxxopts::value< std::string >()->default_value( "" ), "file" )
    ( "format", "Force the input format: auto, coco or dive",
      ::cxxopts::value< std::string >()->default_value( "auto" ), "name" )
    ( "validate", "Check file structure and report problems; makes no changes",
      ::cxxopts::value< bool >()->default_value( "false" ) )
    ;
}

// =======================================================================================
int
json_applet
::run()
{
  kv::logger_handle_t logger = kv::get_logger( "viame.tools.json" );

  auto& cmd_args = command_args();

  if( cmd_args[ "help" ].as< bool >() )
  {
    std::cout << "Usage: viame json [options]\n"
              << "\nPerform filtering and analysis actions on DIVE and COCO JSON files.\n"
              << "\nThis tool mirrors 'viame csv' for JSON annotation files: frame ID\n"
              << "adjustment, type filtering and replacement, track renumbering,\n"
              << "statistics, and structural validation. The format is detected from\n"
              << "the file contents.\n"
              << m_cmd_options->help()
              << "\nExamples:\n"
              << "  viame json -i tracks.json --print-types --track-count\n"
              << "  viame json -i annotations/ --conf-threshold 0.5 --filter-single\n"
              << "  viame json -i \"*.coco.json\" --replace-file synonyms.csv\n"
              << "  viame json -i tracks.json --validate\n"
              << std::endl;
    return EXIT_SUCCESS;
  }

  std::string opt_input = cmd_args[ "input" ].as< std::string >();
  bool opt_decrease_fid = cmd_args[ "decrease-fid" ].as< bool >();
  bool opt_increase_fid = cmd_args[ "increase-fid" ].as< bool >();
  bool opt_assign_uid = cmd_args[ "assign-uid" ].as< bool >();
  bool opt_filter_single = cmd_args[ "filter-single" ].as< bool >();
  bool opt_print_types = cmd_args[ "print-types" ].as< bool >();
  bool opt_caps_only = cmd_args[ "caps-only" ].as< bool >();
  bool opt_track_count = cmd_args[ "track-count" ].as< bool >();
  bool opt_counts_per_frame = cmd_args[ "counts-per-frame" ].as< bool >();
  bool opt_average_box_size = cmd_args[ "average-box-size" ].as< bool >();
  double opt_conf_threshold = cmd_args[ "conf-threshold" ].as< double >();
  double opt_type_threshold = cmd_args[ "type-threshold" ].as< double >();
  bool opt_print_filtered = cmd_args[ "print-filtered" ].as< bool >();
  bool opt_print_single = cmd_args[ "print-single" ].as< bool >();
  int opt_lower_fid = cmd_args[ "lower-fid" ].as< int >();
  int opt_upper_fid = cmd_args[ "upper-fid" ].as< int >();
  std::string opt_replace_file = cmd_args[ "replace-file" ].as< std::string >();
  bool opt_print_fps = cmd_args[ "print-fps" ].as< bool >();
  std::string opt_comp_file = cmd_args[ "comp-file" ].as< std::string >();
  std::string opt_format = cmd_args[ "format" ].as< std::string >();
  bool opt_validate = cmd_args[ "validate" ].as< bool >();

  if( opt_input.empty() )
  {
    std::cout << "No valid input files provided, exiting." << std::endl;
    return EXIT_SUCCESS;
  }

  if( opt_format != "auto" && opt_format != "coco" && opt_format != "dive" )
  {
    LOG_ERROR( logger, "Unknown format: " << opt_format << ", expected auto, coco or dive" );
    return EXIT_FAILURE;
  }

  std::vector< std::string > input_files;

  if( does_folder_exist( opt_input ) )
  {
    list_files_in_folder( opt_input, input_files, true, { ".json" } );
  }
  else if( opt_input.find( '*' ) != std::string::npos )
  {
    input_files = glob_files( opt_input );
  }
  else
  {
    input_files.push_back( opt_input );
  }

  if( opt_caps_only )
  {
    opt_print_types = true;
  }

  if( opt_print_single )
  {
    opt_track_count = true;
  }

  const bool write_output = !opt_validate &&
    ( opt_filter_single || opt_increase_fid || opt_decrease_fid || opt_assign_uid ||
      !opt_replace_file.empty() || opt_lower_fid > 0 || opt_upper_fid > 0 );

  const bool frame_op = opt_increase_fid || opt_decrease_fid ||
    opt_lower_fid > 0 || opt_upper_fid > 0;

  int id_counter = 1;
  std::map< std::string, int > type_counts;
  std::map< std::string, double > type_sizes;
  std::map< std::string, std::map< std::string, std::set< std::string > > > type_ids;
  name_map repl_dict;

  int track_counter = 0;
  int state_counter = 0;
  bool validation_failed = false;

  if( !opt_replace_file.empty() )
  {
    if( !load_replacement_file( opt_replace_file, repl_dict ) )
    {
      std::cout << "Replace file: " << opt_replace_file << " does not exist" << std::endl;
      return EXIT_FAILURE;
    }
  }

  for( const auto& input_file : input_files )
  {
    if( !opt_print_single && !opt_validate )
    {
      if( opt_counts_per_frame )
      {
        std::cout << "# " << get_filename_no_path( input_file ) << std::endl;
      }
      else if( opt_print_fps )
      {
        std::cout << input_file << ",";
      }
      else
      {
        std::cout << "Processing " << input_file << std::endl;
      }
    }

    json_doc doc = load_document( input_file );

    std::string format = opt_format;

    if( format == "auto" )
    {
      format = detect_format( doc );

      if( format == "dive_v1" )
      {
        LOG_INFO( logger, input_file << ": upgrading version 1 DIVE annotations to version 2" );
        upgrade_dive_v1( doc );
        format = "dive";
      }
    }

    if( format != "dive" && format != "coco" )
    {
      throw std::runtime_error( input_file +
        ": could not determine the annotation format, use --format" );
    }

    std::unique_ptr< format_adapter > adapter;

    if( format == "dive" )
    {
      adapter.reset( new dive_adapter( input_file ) );
    }
    else
    {
      adapter.reset( new coco_adapter( input_file ) );
    }

    if( opt_validate )
    {
      const auto messages = adapter->validate( doc );

      for( const auto& message : messages )
      {
        std::cout << message << std::endl;
      }

      if( messages.empty() )
      {
        std::cout << input_file << ": OK" << std::endl;
      }
      else
      {
        validation_failed = true;
      }

      continue;
    }

    std::set< std::string > printed_ids;
    std::vector< entry > entries = adapter->extract( doc );

    auto report_filtered = [&]( const entry& item )
      {
        if( opt_print_filtered && printed_ids.find( item.track_id ) == printed_ids.end() )
        {
          std::cout << "Id: " << item.track_id << " filtered" << std::endl;
          printed_ids.insert( item.track_id );
        }
      };

    if( opt_conf_threshold > 0 )
    {
      drop_set dropped;

      for( const auto& item : entries )
      {
        if( item.confidence < opt_conf_threshold )
        {
          dropped.insert( std::make_pair( item.track_key, item.index ) );
          report_filtered( item );
        }
      }

      if( !dropped.empty() )
      {
        adapter->remove( doc, dropped );
        entries = adapter->extract( doc );
      }
    }

    if( frame_op )
    {
      adapter->require_frames( doc );

      if( opt_decrease_fid )
      {
        adapter->shift_frames( doc, -1 );
      }

      if( opt_increase_fid )
      {
        adapter->shift_frames( doc, 1 );
      }

      if( opt_lower_fid > 0 || opt_upper_fid > 0 )
      {
        const int upper = opt_upper_fid > 0
          ? opt_upper_fid : std::numeric_limits< int >::max();
        adapter->filter_frame_range( doc, opt_lower_fid, upper );
      }

      entries = adapter->extract( doc );
    }

    if( opt_type_threshold > 0 )
    {
      drop_set dropped;

      for( const auto& item : entries )
      {
        if( item.top_score < opt_type_threshold )
        {
          dropped.insert( std::make_pair( item.track_key, item.index ) );
          report_filtered( item );
        }
      }

      if( !dropped.empty() )
      {
        adapter->remove( doc, dropped );
        entries = adapter->extract( doc );
      }
    }

    std::map< std::string, int > states_per_track;

    for( const auto& item : entries )
    {
      states_per_track[ item.track_id ]++;
    }

    bool has_non_single = false;

    for( const auto& counted : states_per_track )
    {
      if( counted.second > 1 )
      {
        has_non_single = true;
        break;
      }
    }

    if( opt_filter_single )
    {
      drop_set dropped;

      for( const auto& item : entries )
      {
        if( states_per_track[ item.track_id ] <= 1 )
        {
          dropped.insert( std::make_pair( item.track_key, item.index ) );
        }
      }

      if( !dropped.empty() )
      {
        adapter->remove( doc, dropped );
        entries = adapter->extract( doc );
      }
    }

    std::set< std::string > unique_ids;
    std::map< std::string, std::set< std::string > > seq_ids;
    std::map< std::pair< int, std::string >, std::map< std::string, int > > frame_counts;

    for( const auto& item : entries )
    {
      unique_ids.insert( item.track_id );

      if( opt_track_count )
      {
        state_counter++;
      }

      if( item.top_type.empty() )
      {
        continue;
      }

      if( opt_print_types || opt_average_box_size )
      {
        type_counts[ item.top_type ]++;

        if( opt_track_count )
        {
          seq_ids[ item.top_type ].insert( item.track_id );
        }
      }

      if( opt_counts_per_frame )
      {
        frame_counts[ std::make_pair( item.frame, item.image_key ) ][ item.top_type ]++;
      }

      if( opt_average_box_size )
      {
        type_sizes[ item.top_type ] +=
          ( item.x2 - item.x1 ) * ( item.y2 - item.y1 );
      }
    }

    if( !seq_ids.empty() )
    {
      type_ids[ input_file ] = seq_ids;
    }

    if( opt_print_fps )
    {
      const double rate = adapter->fps( doc );

      if( rate > 0 )
      {
        std::cout << rate << std::endl;
      }
      else
      {
        std::cout << "unlisted" << std::endl;
      }
    }

    if( opt_track_count )
    {
      track_counter += static_cast< int >( unique_ids.size() );
    }

    if( ( opt_assign_uid || opt_filter_single ) && !has_non_single )
    {
      std::cout << "Sequence " << input_file << " has all single states" << std::endl;
    }

    if( opt_print_single && !has_non_single )
    {
      if( unique_ids.empty() )
      {
        std::cout << "Sequence " << input_file << " contains no detections" << std::endl;
      }
      else
      {
        std::cout << "Sequence " << input_file << " contains only detections" << std::endl;
      }
    }

    if( !opt_replace_file.empty() )
    {
      adapter->replace_types( doc, repl_dict );
    }

    if( opt_assign_uid )
    {
      name_map id_map;

      for( const auto& item : entries )
      {
        if( id_map.find( item.track_id ) == id_map.end() )
        {
          id_map[ item.track_id ] = std::to_string( id_counter++ );
        }
      }

      adapter->renumber_tracks( doc, id_map );
    }

    if( opt_counts_per_frame )
    {
      for( const auto& counted : frame_counts )
      {
        std::string line = counted.first.second;

        for( const auto& per_type : counted.second )
        {
          line += ", " + per_type.first + "=" + std::to_string( per_type.second );
        }

        std::cout << line << std::endl;
      }
    }

    if( write_output )
    {
      save_document( input_file, doc );
    }
  }

  if( opt_validate )
  {
    return validation_failed ? EXIT_FAILURE : EXIT_SUCCESS;
  }

  if( opt_track_count )
  {
    std::cout << "Track count: " << track_counter
              << " , states = " << state_counter << std::endl;
  }

  if( opt_print_types )
  {
    std::cout << "\nTypes found in files:\n" << std::endl;

    auto count_type = [&type_ids]( const std::string& type_name ) -> int
      {
        int count = 0;

        for( const auto& per_file : type_ids )
        {
          const auto& seq_ids = per_file.second;

          if( seq_ids.find( type_name ) != seq_ids.end() )
          {
            count += static_cast< int >( seq_ids.at( type_name ).size() );
          }
        }

        return count;
      };

    for( const auto& counted : type_counts )
    {
      const std::string& type_name = counted.first;

      if( opt_caps_only && !has_uppercase( type_name ) )
      {
        continue;
      }

      if( opt_track_count )
      {
        std::cout << type_name << " " << count_type( type_name ) << std::endl;
      }
      else
      {
        std::cout << type_name << std::endl;
      }
    }
  }

  if( !opt_comp_file.empty() )
  {
    std::ofstream fout( opt_comp_file );

    if( fout )
    {
      fout << "file_name";

      for( const auto& counted : type_counts )
      {
        fout << ", " << counted.first;
      }
      fout << "\n";

      for( const auto& per_file : type_ids )
      {
        fout << per_file.first;

        for( const auto& counted : type_counts )
        {
          const std::string& type_name = counted.first;

          if( per_file.second.find( type_name ) != per_file.second.end() )
          {
            fout << ", " << per_file.second.at( type_name ).size();
          }
          else
          {
            fout << ", 0";
          }
        }
        fout << "\n";
      }
      fout.close();
    }
    else
    {
      std::cerr << "Could not write comparison file: " << opt_comp_file << std::endl;
    }
  }

  if( opt_average_box_size )
  {
    std::cout << "Type - Average Box Area - Total Count" << std::endl;

    for( const auto& sized : type_sizes )
    {
      const double average = sized.second / type_counts[ sized.first ];

      std::cout << sized.first << " "
                << std::setprecision( 17 ) << average << " "
                << type_counts[ sized.first ] << std::endl;
    }
  }

  return EXIT_SUCCESS;
}

} // namespace tools
} // namespace viame
