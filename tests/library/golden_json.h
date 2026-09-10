/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Just enough JSON to read a recording, for the C++ golden tests
///
/// The recordings under `tests/golden/` that a C++ test reads -- the Eigen
/// one and the OpenCV one -- are objects of numbers and flat arrays of
/// numbers, and nothing more. Pulling a parser in for that shape would be a
/// dependency added to the tests of a branch whose point is removing them.
///
/// The recorders keep to the shape this reads: no nested objects inside a
/// case, so a value that would have been `input.data` is `input_data`.

#ifndef VIAME_TESTS_GOLDEN_JSON_H
#define VIAME_TESTS_GOLDEN_JSON_H

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace viame {
namespace testing {

// ----------------------------------------------------------------------------
class golden_json
{
public:
  explicit golden_json( std::string const& path )
  {
    std::ifstream in( path );
    std::stringstream buffer;
    buffer << in.rdbuf();
    text_ = buffer.str();

    if( text_.empty() )
    {
      ADD_FAILURE() << "could not read " << path;
    }
  }

  /// The objects inside the top-level array called \p name.
  std::vector< std::string > section( std::string const& name ) const
  {
    std::vector< std::string > out;
    std::string const key = "\"" + name + "\"";
    size_t at = text_.find( key );

    if( at == std::string::npos )
    {
      return out;
    }

    at = text_.find( '[', at );
    int depth = 0;
    size_t start = 0;

    for( size_t i = at; i < text_.size(); ++i )
    {
      char const ch = text_[ i ];

      if( ch == '{' )
      {
        if( depth == 0 ) { start = i; }
        ++depth;
      }
      else if( ch == '}' )
      {
        if( --depth == 0 )
        {
          out.push_back( text_.substr( start, i - start + 1 ) );
        }
      }
      else if( ch == ']' && depth == 0 )
      {
        break;
      }
    }

    return out;
  }

  /// The string value of \p key in \p object, without its quotes.
  static std::string text( std::string const& object, std::string const& key )
  {
    size_t at = object.find( "\"" + key + "\"" );
    EXPECT_NE( at, std::string::npos ) << key;

    at = object.find( ':', at ) + 1;
    at = object.find( '"', at ) + 1;
    auto const end = object.find( '"', at );

    return object.substr( at, end - at );
  }

  static double number( std::string const& object, std::string const& key )
  {
    size_t at = object.find( "\"" + key + "\"" );
    EXPECT_NE( at, std::string::npos ) << key;

    at = object.find( ':', at ) + 1;
    return std::strtod( object.c_str() + at, nullptr );
  }

  static std::vector< double > numbers( std::string const& object,
                                        std::string const& key )
  {
    std::vector< double > out;
    size_t at = object.find( "\"" + key + "\"" );
    EXPECT_NE( at, std::string::npos ) << key;

    at = object.find( '[', at ) + 1;
    size_t const end = object.find( ']', at );

    char const* p = object.c_str() + at;
    char const* stop = object.c_str() + end;

    while( p < stop )
    {
      char* next = nullptr;
      double const value = std::strtod( p, &next );

      if( next == p ) { break; }

      out.push_back( value );
      p = next;

      while( p < stop && ( *p == ',' || *p == ' ' || *p == '\n' ) ) { ++p; }
    }

    return out;
  }

private:
  std::string text_;
};

} // namespace testing
} // namespace viame

#endif
