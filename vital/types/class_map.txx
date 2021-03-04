// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "class_map.h"

#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>

namespace kwiver {

namespace vital {

// Master list of all type names, and members associated with the same
template < typename T >
signal< std::string const& >
class_map< T >
::class_name_added;

template < typename T >
std::unordered_set< std::string >
class_map< T >
::s_master_name_set;

template < typename T >
std::mutex
class_map< T >
::s_table_mutex;

namespace {

// ----------------------------------------------------------------------------
template < typename T1, typename T2 >
struct less_second
{
  typedef std::pair< T1, T2 > type;
  bool operator()( type const& a, type const& b ) const
  {
    return a.second < b.second;
  }
};

// ----------------------------------------------------------------------------
template < typename T1, typename T2 >
struct more_second
{
  typedef std::pair< T1, T2 > type;
  bool operator()( type const& a, type const& b ) const
  {
    return a.second > b.second;
  }
};

} // namespace <anonymous>

// ----------------------------------------------------------------------------
template < typename T >
class_map< T >
::class_map()
{ }

// ----------------------------------------------------------------------------
template < typename T >
class_map< T >
::class_map( const std::vector< std::string >& class_names,
                        const std::vector< double >& scores )
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

// ----------------------------------------------------------------------------
template < typename T >
class_map< T >
::class_map( const std::string& class_name, double score )
{
  if ( class_name.empty() )
  {
    throw std::invalid_argument( "Must supply a non-empty class name." );
  }

  set_score( class_name, score );
}

// ----------------------------------------------------------------------------
template < typename T >
bool
class_map< T >
::has_class_name( const std::string& class_name ) const
{
  try
  {
    const std::string* str_ptr = find_string( class_name );
    return ( 0 != m_classes.count( str_ptr ) );
  }
  catch ( ... ) {}

  return false;
}

// ----------------------------------------------------------------------------
template < typename T >
double
class_map< T >
::score( const std::string& class_name ) const
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

// ----------------------------------------------------------------------------
template < typename T >
void
class_map< T >
::get_most_likely( std::string& max_name ) const
{
  if ( m_classes.empty() )
  {
    // Throw error
    throw std::runtime_error( "This detection has no scores." );
  }

  auto it = std::max_element( m_classes.begin(), m_classes.end(), less_second< const std::string*, double > () );

  max_name = std::string ( *(it->first) );
}

// ----------------------------------------------------------------------------
template < typename T >
void
class_map< T >
::get_most_likely( std::string& max_name, double& max_score ) const
{
  if ( m_classes.empty() )
  {
    // Throw error
    throw std::runtime_error( "This detection has no scores." );
  }

  auto it = std::max_element( m_classes.begin(), m_classes.end(),
                              less_second< std::string const*, double > () );

  max_name = std::string ( *(it->first) );
  max_score = it->second;
}

// ----------------------------------------------------------------------------
template < typename T >
void
class_map< T >
::set_score( const std::string& class_name, double score )
{
  // Check to see if class_name is in the master set.
  // If not, add it
  std::lock_guard< std::mutex > lock{ s_table_mutex };
  auto it = s_master_name_set.find( class_name );
  if ( it == s_master_name_set.end() )
  {
    auto result = s_master_name_set.insert( class_name );
    class_name_added( class_name );
    it = result.first;
  }

  // Resolve string to canonical pointer
  const std::string* str_ptr = &(*it);

  // Insert new entry into map
  m_classes[str_ptr] = score;
}

// ----------------------------------------------------------------------------
template < typename T >
void
class_map< T >
::delete_score( const std::string& class_name )
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

// ----------------------------------------------------------------------------
template < typename T >
std::vector< std::string >
class_map< T >
::class_names( double threshold ) const
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

// ----------------------------------------------------------------------------
template < typename T >
size_t
class_map< T >
::size() const
{
  return m_classes.size();
}

// ----------------------------------------------------------------------------
template < typename T >
typename class_map< T >::class_const_iterator_t
class_map< T >
::begin() const
{
  return m_classes.begin();
}

// ----------------------------------------------------------------------------
template < typename T >
typename class_map< T >::class_const_iterator_t
class_map< T >
::cbegin() const
{
  return m_classes.cbegin();
}

// ----------------------------------------------------------------------------
template < typename T >
typename class_map< T >::class_const_iterator_t
class_map< T >
::end() const
{
  return m_classes.end();
}

// ----------------------------------------------------------------------------
template < typename T >
typename class_map< T >::class_const_iterator_t
class_map< T >
::cend() const
{
  return m_classes.cend();
}

// ----------------------------------------------------------------------------
/**
 * @brief Resolve string to pointer.
 *
 * This method resolves the supplied string to a pointer to the
 * canonical version in the master set. This is needed because the
 * class_names in this class refer to these strings by address, so we
 * need an address to look up in the map.
 *
 * @param str String to resolve
 *
 * @return Address of string in master list.
 *
 * @throws std::runtime_error if the string is not in the global set.
 */
template < typename T >
const std::string*
class_map< T >
::find_string( const std::string& str ) const
{
  std::lock_guard< std::mutex > lock{ s_table_mutex };
  auto it = s_master_name_set.find( str );
  if ( it == s_master_name_set.end() )
  {
    // Name not associated with any object
    std::stringstream sstr;
    sstr << "Class name \"" << str << "\" is not associated with any object";
    throw std::runtime_error( sstr.str() );
  }

  return &(*it);
}

// ----------------------------------------------------------------------------
template < typename T >
std::vector< std::string >
class_map< T >
::all_class_names()
{
  auto out = []() -> std::vector< std::string >
  {
    std::lock_guard< std::mutex > lock{ s_table_mutex };
    return { s_master_name_set.begin(), s_master_name_set.end() };
  }();

  std::sort( out.begin(), out.end() );
  return out;
}

// ----------------------------------------------------------------------------
template < typename T >
constexpr double
class_map< T >
::INVALID_SCORE;

} // namespace vital

} // namespace kwiver
