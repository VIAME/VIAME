// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "string.h"

#include <unordered_set>
#include <algorithm>
#include <iterator>
#include <memory>    // For std::unique_ptr
#include <cstring>
#include <sstream>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
std::string
string_format( const std::string fmt_str, ... )
{
  int final_n;
  int n = static_cast<int>(fmt_str.size()) * 2; /* Reserve two times as much as the length of the fmt_str */
  std::string str;
  std::unique_ptr< char[] > formatted;
  va_list ap;

  while ( 1 )
  {
    formatted.reset( new char[n] );   /* Wrap the plain char array into the unique_ptr */
    strcpy( &formatted[0], fmt_str.c_str() );
    va_start( ap, fmt_str );
    final_n = vsnprintf( &formatted[0], n, fmt_str.c_str(), ap );
    va_end( ap );
    if ( ( final_n < 0 ) || ( final_n >= n ) )
    {
      n += abs( final_n - n + 1 );
    }
    else
    {
      break;
    }
  }

  return std::string( formatted.get() );
}

// ------------------------------------------------------------------
std::string
join( const std::vector< std::string >& elements,
      const std::string&                str_separator )
{
  const char* const separator = str_separator.c_str();

  switch ( elements.size() )
  {
  case 0:
    return "";

  case 1:
    return elements[0];

  default:
  {
    std::ostringstream os;
    std::copy( elements.cbegin(), elements.cend() - 1,
               std::ostream_iterator< std::string > ( os, separator ) );
    os << *elements.crbegin();
    return os.str();
  }
  } // end switch
}

// ------------------------------------------------------------------
std::string
join( const std::set< std::string >&  elements,
      const std::string&              str_separator )
{
  std::vector< std::string > vec_elem( elements.size() );
  std::copy( elements.begin(), elements.end(), vec_elem.begin() );

  return join( vec_elem, str_separator );
}

//----------------------------------------------------------------------------
/**
 * Removes duplicate strings in a vector while preserving original order
 */
void
erase_duplicates(std::vector<std::string>& items)
{
  // For each item: if it has been seen before, then remove it (inplace)
  std::unordered_set<std::string> seen;
  std::vector<std::string>::iterator itr = items.begin();
  while (itr != items.end())
  {
    if (!seen.insert((*itr)).second)
    {
      // If the item has been seen before, remove it and move itr to the new
      // location of the next item
      itr = items.erase(itr);
    }
    else
    {
      // otherwise go to the next item
      ++itr;
    }
  }
}

}} // end namespace
