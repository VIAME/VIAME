/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// Behaviour of the polygon to mask conversion the VIAME CSV readers use.
///
/// Written against the vgl backed implementation so that the in-house
/// replacement can be held to the same results. Every expectation is a shape
/// small enough to reason about by hand.

#include <convert_polygons_to_mask.h>

#include <viame/core_types/bounding_box.h>
#include <viame/core_types/image.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
/// A polygon in the string form the VIAME CSV carries: "(poly) x y x y ...".
std::string
poly( std::vector< int > const& coordinates )
{
  std::string result = "(poly)";

  for( auto value : coordinates )
  {
    result += " " + std::to_string( value );
  }

  return result;
}

// ----------------------------------------------------------------------------
/// The mask as rows of '.' and '#', which is far easier to read in a failure
/// than a pile of numbers.
std::string
render( kv::image_of< uint8_t > const& mask )
{
  std::string result;

  for( size_t j = 0; j < mask.height(); ++j )
  {
    for( size_t i = 0; i < mask.width(); ++i )
    {
      result += mask( i, j ) ? '#' : '.';
    }
    result += '\n';
  }

  return result;
}

// ----------------------------------------------------------------------------
kv::image_of< uint8_t >
convert( std::vector< std::string > const& polygons,
         double x0, double y0, double x1, double y1 )
{
  kv::image_of< uint8_t > mask;
  viame::convert_polys_to_mask(
    polygons, kv::bounding_box_d( x0, y0, x1, y1 ), mask );
  return mask;
}

} // namespace

// ----------------------------------------------------------------------------
int
main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST ( convert_polygons_to_mask, no_polygons_leaves_the_mask_alone )
{
  kv::image_of< uint8_t > mask;
  viame::convert_polys_to_mask( {}, kv::bounding_box_d( 0, 0, 4, 4 ), mask );

  EXPECT_EQ( 0u, mask.width() );
  EXPECT_EQ( 0u, mask.height() );
}

// ----------------------------------------------------------------------------
TEST ( convert_polygons_to_mask, mask_is_the_size_of_the_box )
{
  auto const mask = convert( { poly( { 0, 0, 3, 0, 3, 3, 0, 3 } ) },
                             0, 0, 6, 4 );

  EXPECT_EQ( 6u, mask.width() );
  EXPECT_EQ( 4u, mask.height() );
  EXPECT_EQ( 1u, mask.depth() );
}

// ----------------------------------------------------------------------------
TEST ( convert_polygons_to_mask, a_filled_rectangle )
{
  // Vertices at 0 and 4 fill columns 0 to 4, and at 0 and 3 fill rows 0 to 3:
  // the scan is inclusive at both ends, so the covered area is one wider and
  // one taller than the difference of the coordinates
  auto const mask = convert( { poly( { 0, 0, 4, 0, 4, 3, 0, 3 } ) },
                             0, 0, 6, 4 );

  EXPECT_EQ(
    "#####.\n"
    "#####.\n"
    "#####.\n"
    "#####.\n",
    render( mask ) );
}

// ----------------------------------------------------------------------------
TEST ( convert_polygons_to_mask, coordinates_are_relative_to_the_box )
{
  // The same rectangle, shifted, with the box shifted to match: the mask is
  // identical because the polygon is stored in image coordinates and the
  // conversion subtracts the box origin
  auto const mask = convert( { poly( { 10, 20, 14, 20, 14, 23, 10, 23 } ) },
                             10, 20, 16, 24 );

  EXPECT_EQ(
    "#####.\n"
    "#####.\n"
    "#####.\n"
    "#####.\n",
    render( mask ) );
}

// ----------------------------------------------------------------------------
TEST ( convert_polygons_to_mask, a_triangle )
{
  auto const mask = convert( { poly( { 0, 0, 4, 0, 0, 4 } ) }, 0, 0, 5, 5 );

  EXPECT_EQ(
    "#####\n"
    "####.\n"
    "###..\n"
    "##...\n"
    "#....\n",
    render( mask ) );
}

// ----------------------------------------------------------------------------
TEST ( convert_polygons_to_mask, a_concave_polygon )
{
  // A 'U': two legs joined along the bottom
  auto const mask = convert(
    { poly( { 0, 0, 2, 0, 2, 4, 4, 4, 4, 0, 6, 0, 6, 6, 0, 6 } ) },
    0, 0, 7, 7 );

  EXPECT_EQ(
    "###.###\n"
    "###.###\n"
    "###.###\n"
    "###.###\n"
    "#######\n"
    "#######\n"
    "#######\n",
    render( mask ) );
}

// ----------------------------------------------------------------------------
TEST ( convert_polygons_to_mask, two_polygons_both_contribute )
{
  auto const mask = convert(
    { poly( { 0, 0, 2, 0, 2, 2, 0, 2 } ),
      poly( { 4, 4, 6, 4, 6, 6, 4, 6 } ) },
    0, 0, 7, 7 );

  EXPECT_EQ(
    "###....\n"
    "###....\n"
    "###....\n"
    ".......\n"
    "....###\n"
    "....###\n"
    "....###\n",
    render( mask ) );
}

// ----------------------------------------------------------------------------
TEST ( convert_polygons_to_mask, a_polygon_reaching_past_the_box_is_clipped )
{
  auto const mask = convert( { poly( { -3, -3, 3, -3, 3, 3, -3, 3 } ) },
                             0, 0, 4, 4 );

  EXPECT_EQ(
    "####\n"
    "####\n"
    "####\n"
    "####\n",
    render( mask ) );
}
