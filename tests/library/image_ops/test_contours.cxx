/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// Connected components and contours, against what OpenCV computed.
///
/// The recorded mask has an L, a ring, a diagonal bar of paired pixels, a
/// single pixel and two squares touching only at a corner -- shapes whose
/// bounding box, area, hull and minimum-area rectangle are all different
/// from each other, and one that eight-connectivity joins and
/// four-connectivity does not.
///
/// Contours are matched by their bounding box rather than by index:
/// `findContours` returns them in its own order and this traces in raster
/// order, and which order they come in is not a contract either library
/// documents.

#include <image_ops/contours.h>

#include <viame/core_types/image.h>

#include "../golden_json.h"
#include "golden_image.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

namespace io = viame::image_ops;
namespace kv = kwiver::vital;

using viame::testing::golden_json;
namespace golden_image = viame::testing::golden_image;

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace {

std::string
whole_file()
{
  static std::string const text = []
  {
    std::ifstream in( golden_image::path() );
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
  }();

  return text;
}

/// The recorded mask.
kv::image_of< uint8_t >
shapes()
{
  auto const text = whole_file();

  auto const width =
    static_cast< size_t >( golden_json::number( text, "shapes_width" ) );
  auto const height =
    static_cast< size_t >( golden_json::number( text, "shapes_height" ) );
  auto const data = golden_json::numbers( text, "shapes_data" );

  EXPECT_EQ( width * height, data.size() );

  kv::image_of< uint8_t > out( width, height, 1 );

  size_t at = 0;
  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      out( i, j, 0 ) = static_cast< uint8_t >( data[ at++ ] );
    }
  }

  return out;
}

/// The recorded objects in the top-level "shapes" array.
std::vector< std::string >
shape_cases()
{
  return golden_json( golden_image::path() ).section( "shapes" );
}

std::string
shape_case( std::string const& name )
{
  for( auto const& object : shape_cases() )
  {
    if( golden_json::text( object, "name" ) == name )
    {
      return object;
    }
  }

  ADD_FAILURE() << "no recorded shape case called '" << name << "'";
  return {};
}

/// The objects inside `contours`, which `section` cannot reach because they
/// are nested one deeper than it reads.
std::vector< std::string >
recorded_contours()
{
  auto const object = shape_case( "contours" );
  std::vector< std::string > out;

  auto at = object.find( "\"contours\"" );

  if( at == std::string::npos )
  {
    return out;
  }

  at = object.find( '[', at );
  int depth = 0;
  size_t start = 0;

  for( size_t i = at; i < object.size(); ++i )
  {
    char const ch = object[ i ];

    if( ch == '{' )
    {
      if( depth == 0 ) { start = i; }
      ++depth;
    }
    else if( ch == '}' )
    {
      if( --depth == 0 )
      {
        out.push_back( object.substr( start, i - start + 1 ) );
      }
    }
  }

  return out;
}

io::rect
recorded_bounds( std::string const& object )
{
  auto const values = golden_json::numbers( object, "bounds" );
  EXPECT_EQ( 4u, values.size() );

  io::rect out;
  out.left = static_cast< long >( values[ 0 ] );
  out.top = static_cast< long >( values[ 1 ] );
  out.right = static_cast< long >( values[ 2 ] );
  out.bottom = static_cast< long >( values[ 3 ] );
  return out;
}

} // namespace

// ----------------------------------------------------------------------------
TEST ( contours, component_counts_match_opencv )
{
  auto const mask = shapes();

  for( auto const& pair : { std::make_pair( "components_four",
                                            io::connectivity::FOUR ),
                            std::make_pair( "components_eight",
                                            io::connectivity::EIGHT ) } )
  {
    auto const object = shape_case( pair.first );

    if( object.empty() )
    {
      continue;
    }

    size_t count = 0;
    auto const labels = io::label_components( mask, pair.second, count );

    auto const expected =
      static_cast< size_t >( golden_json::number( object, "components" ) );

    EXPECT_EQ( expected, count ) << pair.first;
  }
}

// ----------------------------------------------------------------------------
/// The labelling itself, not only how many there are: two pixels share a
/// label here exactly when they share one in the recording.
TEST ( contours, component_labelling_matches_opencv )
{
  auto const mask = shapes();

  for( auto const& pair : { std::make_pair( "components_four",
                                            io::connectivity::FOUR ),
                            std::make_pair( "components_eight",
                                            io::connectivity::EIGHT ) } )
  {
    auto const object = shape_case( pair.first );

    if( object.empty() )
    {
      continue;
    }

    auto const recorded = golden_json::numbers( object, "labels" );
    ASSERT_EQ( mask.width() * mask.height(), recorded.size() );

    size_t count = 0;
    auto const labels = io::label_components( mask, pair.second, count );

    // Compared as a partition rather than by label value, since the
    // numbering is an implementation's own business
    std::vector< int > mine_to_theirs( count + 1, -1 );
    std::vector< int > theirs_to_mine( count + 1, -1 );

    size_t at = 0;
    for( size_t j = 0; j < mask.height(); ++j )
    {
      for( size_t i = 0; i < mask.width(); ++i, ++at )
      {
        auto const mine = labels( i, j, 0 );
        auto const theirs = static_cast< int >( recorded[ at ] );

        ASSERT_EQ( mine == 0, theirs == 0 )
          << pair.first << " at (" << i << ", " << j << ")";

        if( mine == 0 )
        {
          continue;
        }

        ASSERT_LE( static_cast< size_t >( theirs ), count ) << pair.first;

        if( mine_to_theirs[ mine ] < 0 )
        {
          mine_to_theirs[ mine ] = theirs;
          theirs_to_mine[ theirs ] = mine;
        }

        EXPECT_EQ( theirs, mine_to_theirs[ mine ] )
          << pair.first << " at (" << i << ", " << j
          << "): two pixels this joins are apart in the recording";
        EXPECT_EQ( mine, theirs_to_mine[ theirs ] )
          << pair.first << " at (" << i << ", " << j
          << "): two pixels the recording joins are apart here";
      }
    }
  }
}

// ----------------------------------------------------------------------------
TEST ( contours, contour_measurements_match_opencv )
{
  auto const mask = shapes();
  auto const mine = io::find_contours( mask );
  auto const recorded = recorded_contours();

  ASSERT_EQ( recorded.size(), mine.size() )
    << "a different number of outer contours";

  size_t matched = 0;

  for( auto const& object : recorded )
  {
    auto const bounds = recorded_bounds( object );

    auto const found = std::find_if(
      mine.begin(), mine.end(),
      [ & ]( std::vector< io::point > const& contour )
      {
        auto const b = io::bounding_rect( contour );
        return b.left == bounds.left && b.top == bounds.top &&
               b.right == bounds.right && b.bottom == bounds.bottom;
      } );

    ASSERT_NE( mine.end(), found )
      << "no contour bounded by (" << bounds.left << ", " << bounds.top
      << ")-(" << bounds.right << ", " << bounds.bottom << ")";

    ++matched;

    auto const area = golden_json::number( object, "area" );
    EXPECT_NEAR( area, io::contour_area( *found ), 1e-9 )
      << "area of the contour at (" << bounds.left << ", " << bounds.top
      << ")";

    auto const hull_points = golden_json::numbers( object, "hull" );
    auto const hull = io::convex_hull( *found );

    EXPECT_EQ( hull_points.size() / 2, hull.size() )
      << "hull size of the contour at (" << bounds.left << ", "
      << bounds.top << ")";

    auto const min_rect_area =
      golden_json::number( object, "min_rect_area" );
    auto const rect = io::min_area_rect( *found );

    // A minimum-area rectangle over integer points is stable in area even
    // where the orientation is ambiguous, so the area is what is compared
    EXPECT_NEAR( min_rect_area, rect.area(),
                 std::max( 0.02, min_rect_area * 0.02 ) )
      << "minimum-area rectangle of the contour at (" << bounds.left << ", "
      << bounds.top << ")";
  }

  EXPECT_EQ( mine.size(), matched );
}

// ----------------------------------------------------------------------------
/// Values a recording cannot give.
TEST ( contours, a_solid_rectangle_has_the_area_of_its_centres )
{
  kv::image_of< uint8_t > mask( 8, 8, 1 );

  for( size_t j = 0; j < mask.height(); ++j )
  {
    for( size_t i = 0; i < mask.width(); ++i )
    {
      mask( i, j, 0 ) = ( i >= 2 && i < 6 && j >= 1 && j < 5 ) ? 255 : 0;
    }
  }

  auto const contours = io::find_contours( mask );
  ASSERT_EQ( 1u, contours.size() );

  auto const bounds = io::bounding_rect( contours[ 0 ] );
  EXPECT_EQ( 2, bounds.left );
  EXPECT_EQ( 1, bounds.top );
  EXPECT_EQ( 6, bounds.right );
  EXPECT_EQ( 5, bounds.bottom );
  EXPECT_EQ( 16, bounds.area() );

  // Four by four pixels, but the polygon through their centres is three by
  // three -- which is what cv::contourArea answers too
  EXPECT_NEAR( 9.0, io::contour_area( contours[ 0 ] ), 1e-9 );
}

// ----------------------------------------------------------------------------
TEST ( contours, an_isolated_pixel_is_a_contour_of_one_point )
{
  kv::image_of< uint8_t > mask( 5, 5, 1 );

  for( size_t j = 0; j < mask.height(); ++j )
  {
    for( size_t i = 0; i < mask.width(); ++i )
    {
      mask( i, j, 0 ) = 0;
    }
  }

  mask( 2, 3, 0 ) = 255;

  auto const contours = io::find_contours( mask );

  ASSERT_EQ( 1u, contours.size() );
  ASSERT_EQ( 1u, contours[ 0 ].size() );
  EXPECT_EQ( 2, contours[ 0 ][ 0 ].i );
  EXPECT_EQ( 3, contours[ 0 ][ 0 ].j );
  EXPECT_DOUBLE_EQ( 0.0, io::contour_area( contours[ 0 ] ) );
}

// ----------------------------------------------------------------------------
/// A ring is one outer contour, not two: `RETR_EXTERNAL` drops the hole.
TEST ( contours, a_ring_has_one_outer_contour )
{
  kv::image_of< uint8_t > mask( 9, 9, 1 );

  for( size_t j = 0; j < mask.height(); ++j )
  {
    for( size_t i = 0; i < mask.width(); ++i )
    {
      auto const inside = ( i >= 1 && i < 8 && j >= 1 && j < 8 );
      auto const hole = ( i >= 3 && i < 6 && j >= 3 && j < 6 );
      mask( i, j, 0 ) = ( inside && !hole ) ? 255 : 0;
    }
  }

  auto const contours = io::find_contours( mask );
  ASSERT_EQ( 1u, contours.size() );

  // `rect` is half open, so the right edge is one past the last column
  auto const bounds = io::bounding_rect( contours[ 0 ] );
  EXPECT_EQ( 1, bounds.left );
  EXPECT_EQ( 8, bounds.right );
  EXPECT_EQ( 7, bounds.width() );
}

// ----------------------------------------------------------------------------
/// Two squares touching at a corner: eight joins them, four does not.
TEST ( contours, connectivity_decides_a_corner_touch )
{
  kv::image_of< uint8_t > mask( 8, 8, 1 );

  for( size_t j = 0; j < mask.height(); ++j )
  {
    for( size_t i = 0; i < mask.width(); ++i )
    {
      auto const first = ( i < 3 && j < 3 );
      auto const second = ( i >= 3 && i < 6 && j >= 3 && j < 6 );
      mask( i, j, 0 ) = ( first || second ) ? 255 : 0;
    }
  }

  size_t four = 0;
  size_t eight = 0;
  io::label_components( mask, io::connectivity::FOUR, four );
  io::label_components( mask, io::connectivity::EIGHT, eight );

  EXPECT_EQ( 2u, four );
  EXPECT_EQ( 1u, eight );
}

// ----------------------------------------------------------------------------
TEST ( contours, a_hull_of_a_square_is_its_corners )
{
  std::vector< io::point > square;

  for( long j = 0; j <= 4; ++j )
  {
    for( long i = 0; i <= 4; ++i )
    {
      square.push_back( io::point{ i, j } );
    }
  }

  auto const hull = io::convex_hull( square );

  // Four corners, collinear points dropped
  EXPECT_EQ( 4u, hull.size() );

  auto const rect = io::min_area_rect( square );
  EXPECT_NEAR( 16.0, rect.area(), 1e-9 );
}

// ----------------------------------------------------------------------------
/// A rectangle at forty-five degrees is where an axis-aligned bounding box
/// and a minimum-area one differ most.
TEST ( contours, min_area_rect_beats_the_bounding_box_on_a_diagonal )
{
  std::vector< io::point > diamond;

  for( long step = 0; step <= 8; ++step )
  {
    diamond.push_back( io::point{ step, step } );
    diamond.push_back( io::point{ step, 8 - step } );
  }

  auto const bounds = io::bounding_rect( diamond );
  auto const rect = io::min_area_rect( diamond );

  EXPECT_EQ( 81, bounds.area() );
  EXPECT_LT( rect.area(), static_cast< double >( bounds.area() ) );
  EXPECT_NEAR( 4.0, rect.centre_i, 1e-9 );
  EXPECT_NEAR( 4.0, rect.centre_j, 1e-9 );
}

// ----------------------------------------------------------------------------
TEST ( contours, refuses_more_than_one_plane )
{
  kv::image_of< uint8_t > colour( 4, 4, 3 );
  size_t count = 0;

  EXPECT_THROW( io::label_components( colour, io::connectivity::EIGHT,
                                      count ),
                std::invalid_argument );
  EXPECT_THROW( io::find_contours( colour ), std::invalid_argument );
}
