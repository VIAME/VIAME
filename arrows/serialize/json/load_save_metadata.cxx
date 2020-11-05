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
      const double value = kwiver::vital::any_cast< double > ( this->item_value );
      archive( CEREAL_NVP( value ) );
      type = "float";
    }
    else if ( trait.is_integral() )
    {
      // bool metadata passes the is_integral() check but cannot be cast
      // to uint64_t
      if ( trait.tag_type() == typeid( bool ) )
      {
        const bool value = kwiver::vital::any_cast< bool > ( this->item_value );
        archive( CEREAL_NVP( value ) );
        type = "boolean";
      }
      // We don't want negative ints to be serialized as large postive unsigned
      // values since this would be confusing for external applications
      else if ( trait.tag_type() == typeid( int ) )
      {
        const int value = kwiver::vital::any_cast< int > ( this->item_value );
        archive( CEREAL_NVP( value ) );
        type = "integer";
      }
      else
      {
        const uint64_t value = kwiver::vital::any_cast< uint64_t > ( this->item_value );
        archive( CEREAL_NVP( value ) );
        type = "unsigned integer";
      }
    }
    else if ( trait.tag_type() == typeid( std::string ) )
    {
      const std::string value = kwiver::vital::any_cast< std::string > ( this->item_value );
      archive( CEREAL_NVP( value ) );
      type = "string";
    }
    else if ( trait.tag_type() == typeid( kwiver::vital::geo_point ) )
    {
      const kwiver::vital::geo_point value = kwiver::vital::any_cast<kwiver::vital::geo_point  > ( this->item_value );
      archive( CEREAL_NVP( value ) );
      type = "geo-point";
    }
    else if ( trait.tag_type() == typeid( kwiver::vital::geo_polygon ) )
    {
      const kwiver::vital::geo_polygon value = kwiver::vital::any_cast<kwiver::vital::geo_polygon  > ( this->item_value );
      archive( CEREAL_NVP( value ) );
      type = "geo-polygon";
    }
    else
    {
      //+ throw something
    }

    // These two items are included to increase readability of the
    // serialized form and are not used when deserializing.
    archive(  ::cereal::make_nvp( "name", trait.name() ) );
    archive(  ::cereal::make_nvp( "type", type ) );
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
      this->item_value = kwiver::vital::any( value );
    }
    else if ( trait.is_integral() )
    {
      // is_integral() returns true for a bool, which needs to be handled differently
      if ( trait.tag_type() == typeid( bool ) )
      {
        bool value;
        archive( CEREAL_NVP( value ) );
        this->item_value = kwiver::vital::any( value );
      }
      else
      {
        uint64_t value;
        archive( CEREAL_NVP( value ) );
        this->item_value = kwiver::vital::any( value );
      }
    }
    else if ( trait.tag_type() == typeid( std::string ) )
    {
      std::string value;
      archive( CEREAL_NVP( value ) );
      this->item_value = kwiver::vital::any( value );
    }
    else if ( trait.tag_type() == typeid( kwiver::vital::geo_point ) )
    {
      kwiver::vital::geo_point value;
      archive( CEREAL_NVP( value ) );
      this->item_value = kwiver::vital::any( value );
    }
    else if ( trait.tag_type() == typeid( kwiver::vital::geo_polygon ) )
    {
      kwiver::vital::geo_polygon value;
      archive( CEREAL_NVP( value ) );
      this->item_value = kwiver::vital::any( value );
    }
    else
    {
      //+ throw something
    }

  } // end load

};

using meta_vect_t = std::vector< meta_item >;

}

namespace cereal {

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::metadata_vector& meta_packets )
{
  // archive( ::cereal::make_nvp( "size", meta_packets.size() ) );

  std::vector<kwiver::vital::metadata> meta_packets_dereferenced;
  for ( const auto& packet : meta_packets )
  {
    meta_packets_dereferenced.push_back(*packet);
  }
  save( archive, meta_packets_dereferenced );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::metadata_vector& meta )
{
  std::vector<kwiver::vital::metadata> meta_packets_dereferenced;
  load( archive, meta_packets_dereferenced );

  for ( const auto& meta_packet : meta_packets_dereferenced )
  {
    meta.push_back( std::make_shared< kwiver::vital::metadata >( meta_packet ) );
  }
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::metadata& packet_map )
{
  meta_vect_t packet_vec;

  // Serialize one metadata collection
  for ( const auto& item : packet_map )
  {
    // element is <tag, any>
    const auto tag = item.first;
    const auto metap = item.second;
    packet_vec.push_back( meta_item { tag, metap->data() } );

  } // end for

  save( archive, packet_vec );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::metadata& packet_map )
{
  meta_vect_t meta_vect; // intermediate form

  // Deserialize the list of elements for one metadata collection
  load( archive, meta_vect );

  // Convert the intermediate form back to a real metadata collection
  for ( const auto & it : meta_vect )
  {
    const auto& trait = meta_traits.find( it.tag );
    packet_map.add( trait.create_metadata_item( it.item_value ) );
  }
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive,
           const kwiver::vital::metadata_map::map_metadata_t& meta_map )
{
  for ( auto const &meta_vec : meta_map) {
    archive( cereal::make_nvp( std::to_string( meta_vec.first ), meta_vec.second ) );
  }
  // TODO see whether `archive( meta_map );` produces the same result
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive,
           kwiver::vital::metadata_map::map_metadata_t& meta_map )
{
  archive( meta_map );
}

} // end namespace
