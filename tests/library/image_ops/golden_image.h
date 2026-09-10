/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Reading the image_ops recording, and comparing against it
///
/// `tests/golden/image_ops/opencv.json` holds, per case, an input image, the
/// image OpenCV produced from it, a tolerance in counts, and a margin of
/// border the comparison skips. Every `image_ops` test that checks against
/// OpenCV goes through this rather than repeating the walk.

#ifndef VIAME_TESTS_IMAGE_OPS_GOLDEN_IMAGE_H
#define VIAME_TESTS_IMAGE_OPS_GOLDEN_IMAGE_H

#include "../golden_json.h"

#include <viame/core_types/image.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace viame {
namespace testing {

namespace golden_image {

// ----------------------------------------------------------------------------
/// The recording's directory, compiled in by CMake and overridable by the
/// environment so that a copy of the tree can be checked against another.
inline std::string
path()
{
  char const* dir = std::getenv( "VIAME_GOLDEN_IMAGE_OPS_DIR" );

  if( !dir )
  {
    dir = VIAME_GOLDEN_IMAGE_OPS_DIR;
  }

  return std::string( dir ) + "/opencv.json";
}

/// Every recorded case, read once.
inline std::vector< std::string > const&
cases()
{
  static std::vector< std::string > const all =
    golden_json( path() ).section( "cases" );
  return all;
}

/// The case called \p name, or an empty string and a failure.
inline std::string
find( std::string const& name )
{
  for( auto const& object : cases() )
  {
    if( golden_json::text( object, "name" ) == name )
    {
      return object;
    }
  }

  ADD_FAILURE() << "no recorded case called '" << name << "'";
  return {};
}

// ----------------------------------------------------------------------------
/// One recorded image, as vital lays it out.
///
/// The recording is (row, column, plane) because that is numpy's order;
/// `image_of` is indexed (column, row, plane), so the walk transposes.
///
/// \p bias is subtracted from every value, which is how a signed result
/// survives the unsigned recording: the recorder adds 32768 and the reader
/// takes it back off.
template < typename T >
kwiver::vital::image_of< T >
read( std::string const& object, std::string const& prefix, double bias = 0.0 )
{
  auto const width =
    static_cast< size_t >( golden_json::number( object, prefix + "_width" ) );
  auto const height =
    static_cast< size_t >( golden_json::number( object, prefix + "_height" ) );
  auto const planes =
    static_cast< size_t >( golden_json::number( object, prefix + "_planes" ) );

  auto const data = golden_json::numbers( object, prefix + "_data" );

  EXPECT_EQ( width * height * planes, data.size() ) << prefix;

  kwiver::vital::image_of< T > out( width, height, planes );

  size_t at = 0;
  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      for( size_t p = 0; p < planes; ++p )
      {
        out( i, j, p ) = static_cast< T >( data[ at++ ] - bias );
      }
    }
  }

  return out;
}

/// The input of the case called \p name.
inline kwiver::vital::image_of< uint8_t >
input( std::string const& name )
{
  return read< uint8_t >( find( name ), "input" );
}

/// The width the case's expectation has, which a resize has to be asked for.
inline size_t
expected_width( std::string const& name )
{
  return static_cast< size_t >(
    golden_json::number( find( name ), "expected_width" ) );
}

inline size_t
expected_height( std::string const& name )
{
  return static_cast< size_t >(
    golden_json::number( find( name ), "expected_height" ) );
}

// ----------------------------------------------------------------------------
/// Compare \p actual against the recording, at the tolerance it states.
///
/// \p bias is the recorder's offset for a signed result, as in `read`.
template < typename T >
void
compare( std::string const& name, kwiver::vital::image_of< T > const& actual,
         double bias )
{
  auto const object = find( name );

  if( object.empty() )
  {
    return;
  }

  auto const expected = read< double >( object, "expected", bias );
  auto const tolerance = golden_json::number( object, "tolerance" );

  // How many pixels of border to skip. OpenCV ran on the whole fixture and
  // the recording is a window of the result, so a kernel that reads its
  // neighbours sees this window's edge where OpenCV saw real pixels.
  auto const margin =
    static_cast< size_t >( golden_json::number( object, "margin" ) );

  ASSERT_EQ( expected.width(), actual.width() ) << name;
  ASSERT_EQ( expected.height(), actual.height() ) << name;
  ASSERT_EQ( expected.depth(), actual.depth() ) << name;
  ASSERT_GT( expected.width(), 2 * margin ) << name;
  ASSERT_GT( expected.height(), 2 * margin ) << name;

  double worst = 0.0;
  double total = 0.0;
  size_t counted = 0;

  for( size_t j = margin; j + margin < expected.height(); ++j )
  {
    for( size_t i = margin; i + margin < expected.width(); ++i )
    {
      for( size_t p = 0; p < expected.depth(); ++p )
      {
        ++counted;

        auto const difference =
          std::abs( static_cast< double >( actual( i, j, p ) ) -
                    expected( i, j, p ) );

        if( difference > worst )
        {
          worst = difference;
        }

        total += difference;

        EXPECT_LE( difference, tolerance )
          << name << " at (" << i << ", " << j << ", " << p << "): "
          << double( actual( i, j, p ) ) << " against recorded "
          << expected( i, j, p );
      }
    }
  }

  ASSERT_GT( counted, 0u ) << name << ": the margin left nothing to compare";

  // Printed rather than asserted on: a case that passes is more useful with
  // its margin visible, because a tolerance never approached could come down.
  std::cout << "[          ] " << name << ": max " << worst << ", mean "
            << total / static_cast< double >( counted ) << " over " << counted
            << " values, tolerance " << tolerance
            << ( margin ? ", margin " + std::to_string( margin ) : "" )
            << std::endl;
}

/// Compare an unsigned result against the recording.
template < typename T >
void
expect_matches( std::string const& name,
                kwiver::vital::image_of< T > const& actual )
{
  compare( name, actual, 0.0 );
}

/// Compare a signed result, which the recorder shifted by 32768.
template < typename T >
void
expect_matches_signed( std::string const& name,
                       kwiver::vital::image_of< T > const& actual )
{
  compare( name, actual, 32768.0 );
}

} // namespace golden_image

} // namespace testing
} // namespace viame

#endif
