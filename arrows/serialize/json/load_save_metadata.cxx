// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <arrows/serialize/json/load_save.h>

#include <vital/types/metadata.h>
#include <vital/types/metadata_map.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/geo_point.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/polygon.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/vector.hpp>
#include <vital/internal/cereal/types/map.hpp>
#include <vital/internal/cereal/types/utility.hpp>

namespace kv = kwiver::vital;

namespace {

// ---- STATIC DATA ----
static ::kwiver::vital::metadata_traits meta_traits;

// ----------------------------------------------------------------------------
struct meta_item
{

  meta_item( kwiver::vital::vital_metadata_tag  t,
             kwiver::vital::any                 a )
    : tag( t )
    , item_value( a )
  { }

  meta_item() {}

  // ---- member data ----
  kwiver::vital::vital_metadata_tag tag; // numeric tag value
  kwiver::vital::any item_value; // corresponding data item

  // ---------------------------------------------
  /*
   * Save a single metadata item
   */
  template < class Archive >
  void save( Archive& archive ) const
  {
    const auto& trait = meta_traits.find( tag );

    archive( CEREAL_NVP( tag ) );

    std::string type;
    // This is a switch on the item data type
    if ( trait.is_floating_point() )
    {
      const double value = kv::any_cast< double > ( this->item_value );
      archive( CEREAL_NVP( value ) );
      type = "float";
    }
    else if ( trait.is_integral() )
    {
      // bool metadata passes the is_integral() check but cannot be cast
      // to uint64_t
      if ( trait.tag_type() == typeid( bool ) )
      {
        const bool value = kv::any_cast< bool > ( this->item_value );
        archive( CEREAL_NVP( value ) );
        type = "boolean";
      }
      // We don't want negative ints to be serialized as large postive unsigned
      // values since this would be confusing for external applications
      else if ( trait.is_signed() )
      {
        const int value = kv::any_cast< int > ( this->item_value );
        archive( CEREAL_NVP( value ) );
        type = "integer";
      }
      else
      {
        const uint64_t value = kv::any_cast< uint64_t > ( this->item_value );
        archive( CEREAL_NVP( value ) );
        type = "unsigned integer";
      }
    }
    else if ( trait.tag_type() == typeid( std::string ) )
    {
      const std::string value =
        kv::any_cast< std::string > ( this->item_value );
      archive( CEREAL_NVP( value ) );
      type = "string";
    }
    else if ( trait.tag_type() == typeid( kv::geo_point ) )
    {
      const kv::geo_point value =
        kv::any_cast< kv::geo_point > ( this->item_value );
      archive( CEREAL_NVP( value ) );
      type = "geo-point";
    }
    else if ( trait.tag_type() == typeid( kv::geo_polygon ) )
    {
      const kv::geo_polygon value =
        kv::any_cast< kv::geo_polygon > ( this->item_value );
      archive( CEREAL_NVP( value ) );
      type = "geo-polygon";
    }
    else
    {
      //+ throw something
    }

    // These two items are included to increase readability of the
    // serialized form and are not used when deserializing.
    archive( ::cereal::make_nvp( "name", trait.name() ) );
    archive( ::cereal::make_nvp( "type", type ) );
  } // end save

  // -------------------------------------------------
  /*
   * Load a single metadata element
   */
  template<class Archive>
  void load( Archive& archive )
  {
    // Get the tag value
    archive( CEREAL_NVP( tag ) );

    // Get associated traits to assist with decoding the data portion
    const auto& trait = meta_traits.find( tag );

    // this is a switch on the element data type
    if ( trait.is_floating_point() )
    {
      double value;
      archive( CEREAL_NVP( value ) );
      this->item_value = kv::any( value );
    }
    else if ( trait.is_integral() )
    {
      // is_integral() returns true for a bool, which needs to be handled differently
      if ( trait.tag_type() == typeid( bool ) )
      {
        bool value;
        archive( CEREAL_NVP( value ) );
        this->item_value = kv::any( value );
      }
      else
      {
        if( trait.is_signed() )
        {
          int value;
          archive( CEREAL_NVP( value ) );
          this->item_value = kv::any( value );
        }
        else
        {
          uint64_t value;
          archive( CEREAL_NVP( value ) );
          this->item_value = kv::any( value );
        }
      }
    }
    else if ( trait.tag_type() == typeid( std::string ) )
    {
      std::string value;
      archive( CEREAL_NVP( value ) );
      this->item_value = kv::any( value );
    }
    else if ( trait.tag_type() == typeid( kv::geo_point ) )
    {
      kv::geo_point value;
      archive( CEREAL_NVP( value ) );
      this->item_value = kv::any( value );
    }
    else if ( trait.tag_type() == typeid( kv::geo_polygon ) )
    {
      kv::geo_polygon value;
      archive( CEREAL_NVP( value ) );
      this->item_value = kv::any( value );
    }
    else
    {
      //+ throw something
    }

  } // end load

};

using meta_vect_t = std::vector< meta_item >;

} // namespace <anonymous>

namespace cereal {

// ----------------------------------------------------------------------------
void save( ::cereal::JSONOutputArchive& archive,
           kwiver::vital::metadata_vector const& meta_packets )
{
  std::vector< kwiver::vital::metadata > meta_packets_dereferenced;
  for( auto const& packet : meta_packets )
  {
    meta_packets_dereferenced.push_back( *packet );
  }
  save( archive, meta_packets_dereferenced );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive,
           kwiver::vital::metadata_vector& meta )
{
  std::vector< kwiver::vital::metadata > meta_packets_dereferenced;
  load( archive, meta_packets_dereferenced );

  for( auto const& meta_packet : meta_packets_dereferenced )
  {
    meta.push_back(
      std::make_shared< kwiver::vital::metadata >( meta_packet ) );
  }
}

// ----------------------------------------------------------------------------
void save( ::cereal::JSONOutputArchive& archive,
           kwiver::vital::metadata const& packet_map )
{
  meta_vect_t packet_vec;

  // Serialize one metadata collection
  for( auto const& item : packet_map )
  {
    // element is <tag, any>
    const auto tag = item.first;
    const auto metap = item.second;
    packet_vec.emplace_back( tag, metap->data() );
  }

  save( archive, packet_vec );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive,
           kwiver::vital::metadata& packet_map )
{
  meta_vect_t meta_vect; // intermediate form

  // Deserialize the list of elements for one metadata collection
  load( archive, meta_vect );

  // Convert the intermediate form back to a real metadata collection
  for( auto const& it : meta_vect )
  {
    auto const& trait = meta_traits.find( it.tag );
    packet_map.add( trait.create_metadata_item( it.item_value ) );
  }
}

// ----------------------------------------------------------------------------
void save( ::cereal::JSONOutputArchive& archive,
           kwiver::vital::metadata_map::map_metadata_t const& meta_map )
{
  archive( make_size_tag( static_cast< size_type >( meta_map.size() ) ) );

  for ( auto const& meta_vec : meta_map )
  {
    archive( make_map_item( meta_vec.first, meta_vec.second ) );
  }
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive,
           kwiver::vital::metadata_map::map_metadata_t& meta_map )
{
  size_type size;
  archive( make_size_tag( size ) );

  meta_map.clear();

  auto hint = meta_map.begin();
  for( size_t i = 0; i < size; ++i )
  {
    kwiver::vital::frame_id_t key;
    kwiver::vital::metadata_vector value;

    archive( make_map_item(key, value) );
    hint = meta_map.emplace_hint( hint, std::move( key ), std::move( value ) );
  }
}

} // namespace cereal
