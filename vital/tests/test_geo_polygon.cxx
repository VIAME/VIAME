// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core geo_polygon class tests
 */

#include <test_eigen.h>
#include <test_gtest.h>

#include <vital/plugin_loader/plugin_manager.h>

#include <vital/config/config_block.h>

#include <vital/types/geo_polygon.h>
#include <vital/types/geodesy.h>
#include <vital/types/polygon.h>

#include <sstream>

using namespace kwiver::vital;

namespace {

// "It's a magical place." -- P.C.
auto const loc_ll = vector_2d{ -149.484444, -17.619482 };
auto const loc_utm = vector_2d{ 236363.98, 8050181.74 };

auto const loc2_ll = vector_2d{ -73.759291, 42.849631 };

auto const loc_rt = vector_2d{ 0.123456789012345678, -0.987654321098765432 };

auto constexpr crs_ll = SRID::lat_lon_WGS84;
auto constexpr crs_utm_6s = SRID::UTM_WGS84_south + 6;

}

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
bool operator==( polygon const& a, polygon const& b )
{
  auto const k = a.num_vertices();
  if ( b.num_vertices() != k )
  {
    return false;
  }

  for ( auto i = decltype(k){ 0 }; i < k; ++i )
  {
    if ( a.at( i ) != b.at( i ) )
    {
      return false;
    }
  }

  return true;
}

// ----------------------------------------------------------------------------
void PrintTo( polygon const& v, ::std::ostream* os )
{
  auto const k = v.num_vertices();
  (*os) << "(polygon with " << k << " vertices)";
  for ( auto i = decltype(k){ 0 }; i < k; ++i )
  {
    (*os) << "\n  " << i << ": " << ::testing::PrintToString( v.at( i ) );
  }
}

} } // end namespace

// ----------------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, default_constructor)
{
  geo_polygon p;
  EXPECT_TRUE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, constructor_polygon)
{
  geo_polygon p{ { loc_ll }, crs_ll };
  EXPECT_FALSE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, assignment)
{
  geo_polygon p;
  geo_polygon const p1{ { loc_ll }, crs_ll };
  geo_polygon const p2;

  // Paranoia-check initial state
  EXPECT_TRUE( p.is_empty() );

  // Check assignment from non-empty geo_polygon
  p = p1;

  EXPECT_FALSE( p.is_empty() );
  EXPECT_EQ( p1.polygon(), p.polygon() );
  EXPECT_EQ( p1.crs(), p.crs() );

  // Check assignment from empty geo_polygon
  p = p2;

  EXPECT_TRUE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, api)
{
  geo_polygon p{ { loc_ll }, crs_ll };

  // Test values of the point as originally constructed
  [=]() {
    ASSERT_EQ( 1, p.polygon().num_vertices() );
    EXPECT_EQ( crs_ll, p.crs() );
    EXPECT_EQ( loc_ll, p.polygon().at( 0 ) );
    EXPECT_EQ( loc_ll, p.polygon( crs_ll ).at( 0 ) );
  }();

  // Modify the location and test the new values
  p.set_polygon( { loc2_ll }, crs_ll );

  [=]() {
    ASSERT_EQ( 1, p.polygon().num_vertices() );
    EXPECT_EQ( crs_ll, p.crs() );
    EXPECT_EQ( loc2_ll, p.polygon().at( 0 ) );
    EXPECT_EQ( loc2_ll, p.polygon( crs_ll ).at( 0 ) );
  }();

  // Modify the location again and test the new values
  p.set_polygon( { loc_utm }, crs_utm_6s );

  [=]() {
    ASSERT_EQ( 1, p.polygon().num_vertices() );
    EXPECT_EQ( crs_utm_6s, p.crs() );
    EXPECT_EQ( loc_utm, p.polygon().at( 0 ) );
    EXPECT_EQ( loc_utm, p.polygon( crs_utm_6s ).at( 0 ) );
  }();

  // Test that the old location is not cached
  try
  {
    EXPECT_NE( loc2_ll, p.polygon( crs_ll ).at( 0 ) )
      << "Changing the location did not clear the location cache";
  }
  catch (...)
  {
    // If no conversion functor is registered, the conversion will fail; that
    // is okay, since we are only checking here that the point isn't still
    // caching the old location, which it isn't if it needed to attempt a
    // conversion
  }
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, conversion)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  geo_polygon p_ll{ { loc_ll }, crs_ll };
  geo_polygon p_utm{ { loc_utm }, crs_utm_6s };

  auto const conv_loc_utm = p_ll.polygon( p_utm.crs() ).at( 0 );
  auto const conv_loc_ll = p_utm.polygon( p_ll.crs() ).at( 0 );

  auto const epsilon_ll_to_utm = ( loc_utm - conv_loc_utm ).norm();
  auto const epsilon_utm_to_ll = ( loc_ll - conv_loc_ll ).norm();

  EXPECT_MATRIX_NEAR( p_ll.polygon().at( 0 ), conv_loc_ll, 1e-7 );
  EXPECT_MATRIX_NEAR( p_utm.polygon().at( 0 ), conv_loc_utm, 1e-2 );
  EXPECT_LT( epsilon_ll_to_utm, 1e-2 );
  EXPECT_LT( epsilon_utm_to_ll, 1e-7 );

  std::cout << "LL->UTM epsilon: " << epsilon_ll_to_utm << std::endl;
  std::cout << "UTM->LL epsilon: " << epsilon_utm_to_ll << std::endl;
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, config_block_io)
{
  static auto constexpr crs = SRID::lat_lon_NAD83;
  static auto const loc1 = vector_2d{ -77.397577, 38.179969 };
  static auto const loc2 = vector_2d{ -77.329127, 38.181347 };
  static auto const loc3 = vector_2d{ -77.327408, 38.127313 };
  static auto const loc4 = vector_2d{ -77.395808, 38.125938 };

  auto const loc = polygon{ { loc1, loc2, loc3, loc4 } };

  auto const config = config_block::empty_config();
  auto const key = config_block_key_t{ "key" };
  auto const value_in = geo_polygon{ loc, crs };

  config->set_value( key, value_in );

  EXPECT_EQ( loc, config->get_value<geo_polygon>( key ).polygon( crs ) );
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, insert_operator_empty)
{
  geo_polygon p_empty;

  std::stringstream str;
  str << p_empty;

  EXPECT_EQ( str.str(), "{ empty }" );
}

// ----------------------------------------------------------------------------
struct roundtrip_test
{
  char const* text;
  vector_2d loc;
  int crs;
};

// ----------------------------------------------------------------------------
void
PrintTo( roundtrip_test const& v, ::std::ostream* os )
{
  (*os) << v.text;
}

// ----------------------------------------------------------------------------
class geo_polygon_roundtrip : public ::testing::TestWithParam<roundtrip_test>
{
};

// ----------------------------------------------------------------------------
TEST_P(geo_polygon_roundtrip, insert_operator)
{
  auto const expected_loc = GetParam().loc;
  auto const expected_crs = GetParam().crs;

  geo_polygon p{ { expected_loc }, expected_crs };

  // Write polygon to stream
  std::stringstream s;
  s << p;

  // Read back values
  double easting, northing;
  int crs;
  std::string dummy;

  s >> dummy; // {
  s >> easting;
  s >> dummy; // /
  s >> northing;
  s >> dummy; // }
  s >> dummy; // @
  s >> crs;

  // Successful round-trip?
  EXPECT_EQ( expected_loc[0], easting );
  EXPECT_EQ( expected_loc[1], northing );
  EXPECT_EQ( expected_crs, crs );
}

// ----------------------------------------------------------------------------
TEST_P(geo_polygon_roundtrip, config_block)
{
  auto const expected_loc = GetParam().loc;
  auto const expected_crs = GetParam().crs;

  auto const config = config_block::empty_config();
  auto const key = config_block_key_t{ "key" };

  config->set_value( key, geo_polygon{ { expected_loc }, expected_crs } );

  auto const value = config->get_value<geo_polygon>( key );
  ASSERT_EQ( expected_crs, value.crs() );

  auto const actual_loc = value.polygon( expected_crs ).get_vertices()[0];

  // Successful round-trip?
  EXPECT_EQ( expected_loc[0], actual_loc[0] );
  EXPECT_EQ( expected_loc[1], actual_loc[1] );
}

// ----------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(
  ,
  geo_polygon_roundtrip,
  ::testing::Values(
      ( roundtrip_test{ "ll", loc_ll, crs_ll } ),
      ( roundtrip_test{ "utm", loc_utm, crs_utm_6s } ),
      ( roundtrip_test{ "2", loc2_ll, crs_ll } ),
      ( roundtrip_test{ "rt", loc2_ll, 12345 } )
  ) );
