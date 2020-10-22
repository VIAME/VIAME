/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <arrows/serialize/json/load_save.h>

#include <vital/types/metadata.h>
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

    // These two items are included to increase readability of the
    // serialized form and are not used when deserializing.
    archive(  ::cereal::make_nvp( "type", item_value.type_name() ) );
    archive(  ::cereal::make_nvp( "name", trait.name() ) );

    // This is a switch on the item data type
    if ( trait.is_floating_point() )
    {
      const double value = kwiver::vital::any_cast< double > ( this->item_value );
      archive( CEREAL_NVP( value ) );
    }
    else if ( trait.is_integral() )
    {
      // bool metadata passes the is_integral() check but cannot be cast
      // to uint64_t
      if ( trait.tag_type() == typeid( bool ) )
      {
        const bool value = kwiver::vital::any_cast< bool > ( this->item_value );
        archive( CEREAL_NVP( value ) );
      }
      else
      {
        const uint64_t value = kwiver::vital::any_cast< uint64_t > ( this->item_value );
        archive( CEREAL_NVP( value ) );
      }
    }
    else if ( trait.tag_type() == typeid( std::string ) )
    {
      const std::string value = kwiver::vital::any_cast< std::string > ( this->item_value );
      archive( CEREAL_NVP( value ) );
    }
    else if ( trait.tag_type() == typeid( kwiver::vital::geo_point ) )
    {
      const kwiver::vital::geo_point value = kwiver::vital::any_cast<kwiver::vital::geo_point  > ( this->item_value );
      archive( CEREAL_NVP( value ) );
    }
    else if ( trait.tag_type() == typeid( kwiver::vital::geo_polygon ) )
    {
      const kwiver::vital::geo_polygon value = kwiver::vital::any_cast<kwiver::vital::geo_polygon  > ( this->item_value );
      archive( CEREAL_NVP( value ) );
    }
    else
    {
      //+ throw something
    }
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
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::metadata_vector& meta )
{
  std::vector<kwiver::vital::metadata> meta_dereferenced;
  for ( const auto& element : meta )
  {
    meta_dereferenced.push_back( *element );
  }

  save( archive, meta_dereferenced );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::metadata_vector& meta )
{
  std::vector< kwiver::vital::metadata > meta_dereferenced;
  load( archive, meta_dereferenced );

  for ( const auto& packet : meta_dereferenced )
  {
    meta.push_back( std::make_shared< kwiver::vital::metadata > ( packet ) );
  }

}



// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::metadata& meta )
{
  meta_vect_t meta_vect;

  // Serialize one metadata collection
  for ( const auto& mi : meta )
  {
    // element is <tag, any>
    const auto tag = mi.first;
    const auto metap = mi.second;

    meta_vect.push_back( meta_item { tag, metap->data() } );

  } // end for

  save( archive, meta_vect );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::metadata& meta )
{
  meta_vect_t meta_vect; // intermediate form

  // Deserialize the list of elements for one metadata collection
  load( archive, meta_vect );

  // Convert the intermediate form back to a real metadata collection
  for ( const auto & it : meta_vect )
  {
    const auto& trait = meta_traits.find( it.tag );
    meta.add( trait.create_metadata_item( it.item_value ) );
  }
}

} // end namespace
