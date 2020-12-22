// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core geodesy tests
 */

#include <vital/types/geodesy.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>

static auto const loc1 = kwiver::vital::vector_2d{ -73.759291,  42.849631 };
static auto const loc2 = kwiver::vital::vector_2d{   4.857878,  45.777158 };
static auto const loc3 = kwiver::vital::vector_2d{ -62.557243,  82.505337 };
static auto const loc4 = kwiver::vital::vector_2d{ -12.150267,  85.407630 };
static auto const loc5 = kwiver::vital::vector_2d{ 166.644316, -77.840078 };
static auto const loc6 = kwiver::vital::vector_2d{ 107.646964, -83.921037 };

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(geodesy, utm_ups_zones)
{
  auto const z1 = kwiver::vital::utm_ups_zone( loc1 );
  EXPECT_EQ( 18, z1.number );
  EXPECT_EQ( true, z1.north );

  auto const z2 = kwiver::vital::utm_ups_zone( loc2 );
  EXPECT_EQ( 31, z2.number );
  EXPECT_EQ( true, z2.north );

  auto const z3 = kwiver::vital::utm_ups_zone( loc3 );
  EXPECT_EQ( 20, z3.number );
  EXPECT_EQ( true, z3.north );

  auto const z4 = kwiver::vital::utm_ups_zone( loc4 );
  EXPECT_EQ( 0, z4.number );
  EXPECT_EQ( true, z4.north );

  auto const z5 = kwiver::vital::utm_ups_zone( loc5 );
  EXPECT_EQ( 58, z5.number );
  EXPECT_EQ( false, z5.north );

  auto const z6 = kwiver::vital::utm_ups_zone( loc6 );
  EXPECT_EQ( 0, z6.number );
  EXPECT_EQ( false, z6.north );
}

// ----------------------------------------------------------------------------
TEST(geodesy, utm_ups_zone_range_error)
{
  EXPECT_THROW(
    kwiver::vital::utm_ups_zone( 0.0, -100.0 ),
    std::range_error );
  EXPECT_THROW(
    kwiver::vital::utm_ups_zone( 0.0, +100.0 ),
    std::range_error );
}

// ----------------------------------------------------------------------------
TEST(geodesy, descriptions)
{
  using namespace kwiver::vital;

  plugin_manager::instance().load_all_plugins();

  // Helper to get description value from key
  auto get = []( geo_crs_description_t const& desc, char const* key )
  {
    auto i = desc.find( key );
    return ( i == desc.end() ? "(not found)" : i->second );
  };

  // Helper to print a description
  auto print = []( char const* name, geo_crs_description_t const& desc )
  {
    std::cout << name << std::endl;
    for ( auto item : desc )
    {
      std::cout << "  " << item.first << ": " << item.second << std::endl;
    }
  };

  // Test WGS84 lat/lon
  auto const& desc_wgs84_ll = geo_crs_description( SRID::lat_lon_WGS84 );
  print( "WGS84 lat/lon", desc_wgs84_ll );

  EXPECT_EQ( "WGS84", get( desc_wgs84_ll, "datum" ) );
  EXPECT_EQ( "WGS84", get( desc_wgs84_ll, "ellipse" ) );
  EXPECT_EQ( "longlat", get( desc_wgs84_ll, "projection" ) );

  // Test NAD83 lat/lon
  auto const& desc_nad83_ll = geo_crs_description( SRID::lat_lon_NAD83 );
  print( "NAD83 lat/lon", desc_nad83_ll );

  EXPECT_EQ( "NAD83", get( desc_nad83_ll, "datum" ) );
  EXPECT_EQ( "GRS80", get( desc_nad83_ll, "ellipse" ) );
  EXPECT_EQ( "longlat", get( desc_nad83_ll, "projection" ) );

  // Test WGS84 UTM North
  constexpr auto WGS84_UTM_21N = SRID::UTM_WGS84_north + 21;
  auto const& desc_wgs84_utm_21n = geo_crs_description( WGS84_UTM_21N );
  print( "WGS84 UTM North 21", desc_wgs84_utm_21n );

  EXPECT_EQ( "WGS84", get( desc_wgs84_utm_21n, "datum" ) );
  EXPECT_EQ( "WGS84", get( desc_wgs84_utm_21n, "ellipse" ) );
  EXPECT_EQ( "utm", get( desc_wgs84_utm_21n, "projection" ) );
  EXPECT_EQ( "21", get( desc_wgs84_utm_21n, "zone" ) );
  EXPECT_EQ( "north", get( desc_wgs84_utm_21n, "hemisphere" ) );

  // Test WGS84 UTM South
  constexpr auto WGS84_UTM_55S = SRID::UTM_WGS84_south + 55;
  auto const& desc_wgs84_utm_55s = geo_crs_description( WGS84_UTM_55S );
  print( "WGS84 UTM South 55", desc_wgs84_utm_55s );

  EXPECT_EQ( "WGS84", get( desc_wgs84_utm_55s, "datum" ) );
  EXPECT_EQ( "WGS84", get( desc_wgs84_utm_55s, "ellipse" ) );
  EXPECT_EQ( "utm", get( desc_wgs84_utm_55s, "projection" ) );
  EXPECT_EQ( "55", get( desc_wgs84_utm_55s, "zone" ) );
  EXPECT_EQ( "south", get( desc_wgs84_utm_55s, "hemisphere" ) );

  // Test NAD83 UTM West
  constexpr auto NAD83_UTM_18S = SRID::UTM_NAD83_northwest + 18;
  auto const& desc_nad83_utm_18s = geo_crs_description( NAD83_UTM_18S );
  print( "NAD83 UTM North 18", desc_nad83_utm_18s );

  EXPECT_EQ( "NAD83", get( desc_nad83_utm_18s, "datum" ) );
  EXPECT_EQ( "GRS80", get( desc_nad83_utm_18s, "ellipse" ) );
  EXPECT_EQ( "utm", get( desc_nad83_utm_18s, "projection" ) );
  EXPECT_EQ( "18", get( desc_nad83_utm_18s, "zone" ) );
  EXPECT_EQ( "north", get( desc_nad83_utm_18s, "hemisphere" ) );

  // Test NAD83 UTM East
  constexpr auto NAD83_UTM_59S = SRID::UTM_NAD83_northeast + 59;
  auto const& desc_nad83_utm_59s = geo_crs_description( NAD83_UTM_59S );
  print( "NAD83 UTM North 59", desc_nad83_utm_59s );

  EXPECT_EQ( "NAD83", get( desc_nad83_utm_59s, "datum" ) );
  EXPECT_EQ( "GRS80", get( desc_nad83_utm_59s, "ellipse" ) );
  EXPECT_EQ( "utm", get( desc_nad83_utm_59s, "projection" ) );
  EXPECT_EQ( "59", get( desc_nad83_utm_59s, "zone" ) );
  EXPECT_EQ( "north", get( desc_nad83_utm_59s, "hemisphere" ) );

  // Test WGS84 UPS North
  auto const& desc_wgs84_ups_n = geo_crs_description( SRID::UPS_WGS84_north );
  print( "WGS84 UPS North", desc_wgs84_ups_n );

  EXPECT_EQ( "WGS84", get( desc_wgs84_ups_n, "datum" ) );
  EXPECT_EQ( "WGS84", get( desc_wgs84_ups_n, "ellipse" ) );
  EXPECT_EQ( "stere", get( desc_wgs84_ups_n, "projection" ) );

  // Test WGS84 UPS South
  auto const& desc_wgs84_ups_s = geo_crs_description( SRID::UPS_WGS84_south );
  print( "WGS84 UPS South", desc_wgs84_ups_s );

  EXPECT_EQ( "WGS84", get( desc_wgs84_ups_s, "datum" ) );
  EXPECT_EQ( "WGS84", get( desc_wgs84_ups_s, "ellipse" ) );
  EXPECT_EQ( "stere", get( desc_wgs84_ups_s, "projection" ) );
}
