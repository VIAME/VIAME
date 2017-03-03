/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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

#include "detected_object_type.h"

#include <vital/vital_foreach.h>

#include <stdexcept>
#include <limits>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace kwiver {
namespace vital {

const double detected_object_type::INVALID_SCORE = std::numeric_limits< double >::min();

// Master list of all class_names
std::set< std::string > detected_object_type::s_master_name_set;


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
detected_object_type::
detected_object_type()
{ }

detected_object_type::
detected_object_type( const std::vector< std::string >& class_names,
                      const std::vector< double >& scores )
{
  if ( class_names.size() != scores.size() )
  {
    // Throw error
    throw std::invalid_argument( "Parameter vector sizes differ." );
  }

  if ( class_names.size() == 0 )
  {
    // Throw error
    throw std::invalid_argument( "Parameter vector are empty." );
  }

  for (size_t i = 0; i < class_names.size(); i++)
  {
    set_score( class_names[i], scores[i] );
  }
}


// ------------------------------------------------------------------
bool
detected_object_type::
has_class_name( const std::string& class_name ) const
{
  return find_string( class_name, false ) != NULL;
}


// ------------------------------------------------------------------
double
detected_object_type::
score( const std::string& class_name ) const
{
  const std::string* str_ptr = find_string( class_name );

  if ( 0 == m_classes.count( str_ptr ) )
  {
    // Name not associated with this object
    std::stringstream sstr;
    sstr << "Class name \"" << class_name << "\" is not associated with this object";
    throw std::runtime_error( sstr.str() );
  }

  auto it = m_classes.find( str_ptr );
  return it->second; // return score
}


// ------------------------------------------------------------------
void
detected_object_type::
  get_most_likely( std::string& max_name, double& max_score ) const
{
  if ( m_classes.empty() )
  {
    // Throw error
    throw std::runtime_error( "This detection has no scores." );
  }

  auto it = std::max_element( m_classes.begin(), m_classes.end(), less_second< const std::string*, double > () );

  max_name = std::string ( *(it->first) );
  max_score = it->second;
}


// ------------------------------------------------------------------
void
detected_object_type::
set_score( const std::string& class_name, double score )
{
  // Check to see if class_name is in the master set.
  // If not, add it
  auto it = s_master_name_set.find( class_name );
  if ( it == s_master_name_set.end() )
  {
    auto result = s_master_name_set.insert( class_name );
    it = result.first;
  }

  // Resolve string to canonical pointer
  const std::string* str_ptr = &(*it);

  // Insert new entry into map
  m_classes[str_ptr] = score;
}


// ------------------------------------------------------------------
void
detected_object_type::
delete_score( const std::string& class_name )
{
  auto str_ptr = find_string( class_name );
  if ( 0 == m_classes.count( str_ptr ) )
  {
    // Name not associated with this object
    std::stringstream sstr;
    sstr << "Class name \"" << class_name << "\" is not associated with this object";
    throw std::runtime_error( sstr.str() );
  }

  m_classes.erase( str_ptr );
}


// ------------------------------------------------------------------
std::vector< std::string >
detected_object_type::
class_names( double threshold ) const
{
  std::vector< std::pair< const std::string*, double > > items( m_classes.begin(), m_classes.end() );

  // sort map by value descending order
  std::sort( items.begin(), items.end(), more_second< const std::string*, double > () );

  std::vector< std::string > list;

  const size_t limit( items.size() );
  for ( size_t i = 0; i < limit; i++ )
  {
    if ( items[i].second < threshold )
    {
      break;
    }

    list.push_back( *(items[i].first) );
  }

  return list;
}


// ------------------------------------------------------------------
size_t
detected_object_type::
size() const
{
  return m_classes.size();
}


// ------------------------------------------------------------------
/**
 * @brief Resolve string to pointer.
 *
 * This method resolves the supplied string to a pointer to the
 * canonical version in the master set. This is needed because the
 * class_names in this class refer to these strings by address, so we
 * need an address to look up in the map. If the string is not found,
 * there's the option to either return no string or an exception.
 *
 * @param str String to resolve
 *
 * @return Address of string in master list.
 */
const std::string*
detected_object_type::
find_string( const std::string& str, bool exception ) const
{
  auto it = s_master_name_set.find( str );
  if ( it == s_master_name_set.end() )
  {
    // Name not associated with any object
    if( exception )
    {
      std::stringstream sstr;
      sstr << "Class name \"" << str << "\" is not associated with any object";

      throw std::runtime_error( sstr.str() );
    }
    else
    {
      return NULL;
    }
  }

  return &(*it);
}


// ------------------------------------------------------------------
std::vector< std::string >
detected_object_type::
all_class_names()
{
  std::vector< std::string > names( s_master_name_set.begin(), s_master_name_set.end() );
  return names;
}

} } // end namespace
