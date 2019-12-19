/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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

#include <vital/types/activity_type.h>


#include <stdexcept>
#include <limits>
#include <sstream>
#include <algorithm>

namespace kwiver {
namespace vital {

// ==================================================================
namespace {

template < typename T1, typename T2 >
struct less_second
{
  typedef std::pair< T1, T2 > type;
  bool operator()( type const& a, type const& b ) const
  {
    return a.second < b.second;
  }
};


template < typename T1, typename T2 >
struct more_second
{
  typedef std::pair< T1, T2 > type;
  bool operator()( type const& a, type const& b ) const
  {
    return a.second > b.second;
  }
};

} // end namespace

// ------------------------------------------------------------------
activity_type
::activity_type()
{ }

activity_type
::activity_type( const std::vector< activity_label_t >& class_names,
                 const std::vector< activity_confidence_t >& scores )
{
  if ( class_names.size() != scores.size() )
  {
    // Throw error
    throw std::invalid_argument( "Parameter vector sizes differ." );
  }

  if ( class_names.empty() )
  {
    // Throw error
    throw std::invalid_argument( "Parameter vector are empty." );
  }

  for ( size_t i = 0; i < class_names.size(); i++ )
  {
    set_score( class_names[i], scores[i] );
  }
}


// ------------------------------------------------------------------
activity_type
::activity_type( const activity_label_t& class_name, activity_confidence_t score )
{
  if ( class_name.empty() )
  {
    throw std::invalid_argument( "Must supply a non-empty class name." );
  }

  set_score( class_name, score );
}


// ------------------------------------------------------------------
bool
activity_type
::has_class_name( const activity_label_t& class_name ) const
{
  return m_classes.find( class_name ) != m_classes.end();
}


// ------------------------------------------------------------------
activity_confidence_t
activity_type
::score( const activity_label_t& class_name ) const
{
  auto it = m_classes.find( class_name );
  if ( it == m_classes.end() )
  {
    // Name not associated with this object
    std::stringstream sstr;
    sstr << "Class name \"" << class_name << "\" is not associated with this object";
    throw std::runtime_error( sstr.str() );
  }
  return it->second; // return score
}


// ------------------------------------------------------------------
const activity_label_t
activity_type
::get_most_likely_class( ) const
{
  if ( m_classes.empty() )
  {
    // Throw error
    throw std::runtime_error( "This activity has no scores." );
  }

  auto it = std::max_element( m_classes.begin(), m_classes.end(),
                less_second< const activity_label_t, activity_confidence_t > () );

  return it->first;
}


// ------------------------------------------------------------------
std::pair< activity_label_t, activity_confidence_t >
activity_type
::get_most_likely_class_and_score( ) const
{
  if ( m_classes.empty() )
  {
    // Throw error
    throw std::runtime_error( "This activity has no scores." );
  }

  auto it = std::max_element( m_classes.begin(), m_classes.end(),
                    less_second< activity_label_t, activity_confidence_t > () );
  std::pair< activity_label_t, activity_confidence_t> most_likely_item( it->first,
                                                                   it->second );
  return most_likely_item;
}


// ------------------------------------------------------------------
void
activity_type
::set_score( const activity_label_t& class_name, activity_confidence_t score )
{
  // Insert new entry into map
  m_classes[class_name] = score;
}


// ------------------------------------------------------------------
void
activity_type
::delete_score( const activity_label_t& class_name )
{
  m_classes.erase( class_name );
}


// ------------------------------------------------------------------
std::vector< std::string >
activity_type
::class_names( activity_confidence_t threshold ) const
{
  std::vector< std::pair< activity_label_t, activity_confidence_t> >
    items( m_classes.begin(), m_classes.end() );

  // sort map by value descending order
  std::sort( items.begin(), items.end(), more_second< activity_label_t, activity_confidence_t > () );

  std::vector< std::string > list;

  for ( auto& item : items )
  {
    if ( item.second < threshold )
    {
      break;
    }

    list.push_back( item.first );
  }
  return list;
}


// ------------------------------------------------------------------
size_t
activity_type
::size() const
{
  return m_classes.size();
}


// ------------------------------------------------------------------
activity_type::class_const_iterator_t
activity_type
::begin() const
{
  return m_classes.begin();
}


// ------------------------------------------------------------------
activity_type::class_const_iterator_t
activity_type
::end() const
{
  return m_classes.end();
}


// ------------------------------------------------------------------
std::vector< activity_label_t >
activity_type
::all_class_names()
{
  std::vector< activity_label_t > names;
  for ( auto& m_class : m_classes )
  {
    names.push_back( m_class.first );
  }
  return names;
}

} } // end namespace
