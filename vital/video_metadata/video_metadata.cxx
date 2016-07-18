/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

/**
 * \file
 * \brief This file contains the implementation for vital video metadata.
 */

#include "video_metadata.h"
#include "video_metadata_traits.h"

#include <vital/util/demangle.h>

namespace kwiver {
namespace vital {

const std::string video_metadata::MISB_0104( "MISB_0104" );
const std::string video_metadata::MISB_0601( "MISB_0601" );


// ------------------------------------------------------------------
// video metadata exception support
video_metadata_exception
::video_metadata_exception( std::string const& str )
{
  m_what = str;
}


video_metadata_exception
::~video_metadata_exception() VITAL_NOTHROW
{ }


// ----------------------------------------------------------------
/*
 * This class is returned when find can not locate the requested tag.
 *
 */
  class unknown_metadata_item
    : public metadata_item
  {
  public:
    // -- CONSTRUCTORS --
    unknown_metadata_item()
      : metadata_item( "Requested metadata item is not in collection", 0 )
    { }

    virtual ~unknown_metadata_item() {}
    virtual vital_metadata_tag tag() const { return static_cast< vital_metadata_tag >(0); }
    virtual std::type_info const& type() const { return typeid( void ); }
    virtual std::string as_string() const { return "--Unknown metadata item--"; }
    virtual double as_double() const { return 0; }
    virtual double as_uint64() const { return 0; }

  }; // end class unknown_metadata_item

// ==================================================================

metadata_item
::metadata_item(std::string name, kwiver::vital::any const& data )
    : m_name( name ),
      m_data( data )
{ }


metadata_item
::~metadata_item()
{ }


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
video_metadata
::video_metadata()
{ }


video_metadata
::~video_metadata()
{

}

void
video_metadata
::add( metadata_item* item )
{
  this->m_metadata_map[item->tag()] = item_ptr(item);
}


bool
video_metadata
::has( vital_metadata_tag tag )
{
  return m_metadata_map.find( tag ) != m_metadata_map.end();
}


// ------------------------------------------------------------------
metadata_item const&
video_metadata
::find( vital_metadata_tag tag )
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
video_metadata
::erase( vital_metadata_tag tag )
{
  return m_metadata_map.erase( tag ) > 0;
}


// ------------------------------------------------------------------
video_metadata::const_iterator_t
video_metadata
::begin() const
{
  return m_metadata_map.begin();
}


video_metadata::const_iterator_t
video_metadata
::end() const
{
  return m_metadata_map.end();
}


size_t
video_metadata
::size() const
{
  return m_metadata_map.size();
}


bool
video_metadata
::empty() const
{
  return m_metadata_map.empty();
}


// ------------------------------------------------------------------
void
video_metadata
::set_timestamp( kwiver::vital::timestamp const& ts )
{
  this->m_timestamp = ts;
}


kwiver::vital::timestamp const&
video_metadata
::timestamp() const
{
  return this->m_timestamp;
}


  // ------------------------------------------------------------------
std::type_info const&
video_metadata
::typeid_for_tag( vital_metadata_tag tag )
{

  switch (tag)
  {
#define VITAL_META_TRAIT_CASE(TAG, NAME, T) case VITAL_META_ ## TAG: return typeid(T);

    KWIVER_VITAL_METADATA_TAGS( VITAL_META_TRAIT_CASE )

#undef VITAL_META_TRAIT_CASE

  default: return typeid(void);
  }
}


// ------------------------------------------------------------------
std::string
video_metadata
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
    char const byte = val[i];
    if ( ! isprint( byte ) )
    {
      ascii.append( 1, '.' );
      unprintable_found = true;
    }
    else
    {
      ascii.append( 1, byte );
    }

    // format as hex
    if (i > 0)
    {
      hex += " ";
    }

    hex += hex_chars[ ( byte & 0xF0 ) >> 4 ];
    hex += hex_chars[ ( byte & 0x0F ) >> 0 ];

  } // end for

  if (unprintable_found)
  {
    ascii += " (" + hex + ")";
  }

  return ascii;
}


// ------------------------------------------------------------------
std::ostream& print_metadata( std::ostream& str, video_metadata const& metadata )
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
       << video_metadata::format_string (ix->second->as_string())
       << std::endl;
  } // end for

  return str;
}

} } // end namespace
