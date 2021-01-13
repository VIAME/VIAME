// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the implementation for vital metadata.
 */

#include "metadata.h"

#include <vital/util/demangle.h>

namespace kwiver {
namespace vital {

metadata_item
::metadata_item( std::string const& p_name,
                 kwiver::vital::any const& p_data,
                 vital_metadata_tag p_tag )
    : m_name{ p_name },
      m_data{ p_data },
      m_tag{ p_tag }
{
}

// ----------------------------------------------------------------------------
metadata_item
::metadata_item( std::string const& p_name,
                 kwiver::vital::any&& p_data,
                 vital_metadata_tag p_tag )
    : m_name{ p_name },
      m_data{ std::move( p_data ) },
      m_tag{ p_tag }
{
}

// ----------------------------------------------------------------------------
bool
metadata_item
::is_valid() const
{
  return true;
}

std::string const&
metadata_item
::name() const
{
  return this->m_name;
}

kwiver::vital::any
metadata_item
::data() const
{
  return this->m_data;
}

double
metadata_item
::as_double() const
{
  return kwiver::vital::any_cast< double > ( this->m_data );
}

bool
metadata_item
::has_double() const
{
  return m_data.type() == typeid( double );
}

uint64_t
metadata_item
::as_uint64() const
{
  return kwiver::vital::any_cast< uint64_t > ( this->m_data );
}

bool
metadata_item
::has_uint64() const
{
  return m_data.type() == typeid( uint64_t );
}

bool
metadata_item
::has_string() const
{
  return m_data.type() == typeid( std::string );
}

// ==================================================================
metadata
::metadata()
{ }

// ---------------------------------------------------------------------
void
metadata
::add( std::unique_ptr< metadata_item >&& item )
{
  if ( !item )
  {
    throw std::invalid_argument{ "null pointer" };
  }

  auto const tag = item->tag();
#ifdef VITAL_STD_MAP_UNIQUE_PTR_ALLOWED
  this->m_metadata_map[ tag ] = std::move( item );
#else
  this->m_metadata_map[ tag ] = item_ptr{ item.release() };
#endif
}

// ---------------------------------------------------------------------
void
metadata
::add_copy( std::shared_ptr<metadata_item const> const& item )
{
  if ( !item )
  {
    throw std::invalid_argument{ "null pointer" };
  }

  // Since the design intent for this map is that the metadata
  // collection owns the elements, we will clone the item passed in.
  // The original parameter will be freed eventually.
  this->m_metadata_map[ item->tag() ] = item_ptr{ item->clone() };
}

// -------------------------------------------------------------------
bool
metadata
::has( vital_metadata_tag tag ) const
{
  return m_metadata_map.find( tag ) != m_metadata_map.end();
}

// ------------------------------------------------------------------
metadata_item const&
metadata
::find( vital_metadata_tag tag ) const
{
  static unknown_metadata_item unknown_item;

  const_iterator_t it = m_metadata_map.find( tag );
  if ( it == m_metadata_map.end() )
  {
    return unknown_item;
  }

  return *(it->second);
}

// ------------------------------------------------------------------
bool
metadata
::erase( vital_metadata_tag tag )
{
  return m_metadata_map.erase( tag ) > 0;
}

// ------------------------------------------------------------------
metadata::const_iterator_t
metadata
::begin() const
{
  return m_metadata_map.begin();
}

metadata::const_iterator_t
metadata
::cbegin() const
{
  return m_metadata_map.cbegin();
}

metadata::const_iterator_t
metadata
::end() const
{
  return m_metadata_map.end();
}

metadata::const_iterator_t
metadata
::cend() const
{
  return m_metadata_map.cend();
}

// ---------------------------------------------------------------------
size_t
metadata
::size() const
{
  return m_metadata_map.size();
}

// ---------------------------------------------------------------------
bool
metadata
::empty() const
{
  return m_metadata_map.empty();
}

// ------------------------------------------------------------------
void
metadata
::set_timestamp( kwiver::vital::timestamp const& ts )
{
  this->m_timestamp = ts;
}

// ---------------------------------------------------------------------
kwiver::vital::timestamp const&
metadata
::timestamp() const
{
  return this->m_timestamp;
}

// ------------------------------------------------------------------
std::string
metadata
::format_string( std::string const& val )
{
  const char hex_chars[16] = { '0', '1', '2', '3', '4', '5', '6', '7',
                               '8', '9', 'A', 'B', 'C', 'D', 'E', 'F' };
  const size_t len( val.size() );
  bool unprintable_found(false);
  std::string ascii;
  std::string hex;

  for (size_t i = 0; i < len; i++)
  {
    char const l_byte = val[i];
    if ( ! isprint( l_byte ) )
    {
      ascii.append( 1, '.' );
      unprintable_found = true;
    }
    else
    {
      ascii.append( 1, l_byte );
    }

    // format as hex
    if (i > 0)
    {
      hex += " ";
    }

    hex += hex_chars[ ( l_byte & 0xF0 ) >> 4 ];
    hex += hex_chars[ ( l_byte & 0x0F ) >> 0 ];

  } // end for

  if (unprintable_found)
  {
    ascii += " (" + hex + ")";
  }

  return ascii;
}

// ------------------------------------------------------------------
std::ostream& print_metadata( std::ostream& str, metadata const& metadata )
{
  auto eix = metadata.end();
  for ( auto ix = metadata.begin(); ix != eix; ix++)
  {
    // process metada items
   std::string name = ix->second->name();
   kwiver::vital::any data = ix->second->data();

   str << "Metadata item: "
       << name
       << " <" << demangle( ix->second->type().name() ) << ">: "
       << metadata::format_string (ix->second->as_string())
       << std::endl;
  } // end for

  return str;
}

// ----------------------------------------------------------------------------
bool test_equal_content( const kwiver::vital::metadata& one,
                         const kwiver::vital::metadata& other )
{
  // They must be the same size to be the same content
  if ( one.size() != other.size() ) { return false; }

  for ( const auto& mi : one )
  {
    // element is <tag, any>
    const auto tag = mi.first;
    const auto metap = mi.second;

    auto& omi = other.find( tag );
    if ( ! omi ) { return false; }

    // It is simpler to just do a string comparison than to try to do
    // a type specific comparison.
    if ( metap->as_string() != omi.as_string() ) { return false; }

  } // end for

  return true;
}

} } // end namespace
