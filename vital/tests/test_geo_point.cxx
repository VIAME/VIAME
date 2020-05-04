/*ckwg +29
 * Copyright 2017, 2019 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

/**
 * \file
 * \brief core geo_point class tests
 */

#include <test_eigen.h>
#include <test_gtest.h>

#include <vital/types/geo_point.h>
#include <vital/types/geodesy.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <sstream>

using namespace kwiver::vital;

namespace {

auto const loc1 = vector_2d{ -73.759291, 42.849631 };
auto const loc2 = vector_2d{ -73.757161, 42.849764 };
auto const loc3 = vector_2d{ 601375.01, 4744863.31 };

auto const loc1a = vector_3d{ -73.759291, 42.849631, 50 };
auto const loc2a = vector_3d{ -73.757161, 42.849764, 50 };
auto const loc3a = vector_3d{ 601375.01, 4744863.31, 50 };

auto const locrt = vector_2d{ 0.123456789012345678, -0.987654321098765432 };

auto constexpr crs_ll = SRID::lat_lon_WGS84;
auto constexpr crs_utm_18n = SRID::UTM_WGS84_north + 18;

}

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(geo_point, default_constructor)
{
  geo_point p;
  EXPECT_TRUE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_point, constructor_point)
{
  geo_point p{ loc1, crs_ll };
  EXPECT_FALSE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_point, assignment)
{
  geo_point p;
  geo_point const p1{ loc1, crs_ll };
  geo_point const p2;

  // Paranoia-check initial state
  EXPECT_TRUE( p.is_empty() );

  // Check assignment from non-empty geo_point
  p = p1;

  EXPECT_FALSE( p.is_empty() );
  EXPECT_EQ( p1.location(), p.location() );
  EXPECT_EQ( p1.crs(), p.crs() );

  // Check assignment from empty geo_point
  p = p2;

  EXPECT_TRUE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_point, api)
{
  geo_point p{ loc1, crs_ll };
  vector_3d _loc1{ loc1[0], loc1[1], 0 };

  // Test values of the point as originally constructed
  EXPECT_EQ( crs_ll, p.crs() );
  EXPECT_EQ( _loc1, p.location() );
  EXPECT_EQ( _loc1, p.location( crs_ll ) );

  // Modify the location and test the new values
  p.set_location( loc3, crs_utm_18n );
  vector_3d _loc3{ loc3[0], loc3[1], 0 };

  EXPECT_EQ( crs_utm_18n, p.crs() );
  EXPECT_EQ( _loc3, p.location() );
  EXPECT_EQ( _loc3, p.location( crs_utm_18n ) );

  // Modify the location again and test the new values
  p.set_location( loc2, crs_ll );
  vector_3d _loc2{ loc2[0], loc2[1], 0 };

  EXPECT_EQ( crs_ll, p.crs() );
  EXPECT_EQ( _loc2, p.location() );
  EXPECT_EQ( _loc2, p.location( crs_ll ) );

  // Test that the old location is not cached
  try
  {
    EXPECT_NE( _loc3, p.location( crs_utm_18n ) )
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
TEST(geo_point, conversion)
{
  plugin_manager::instance().load_all_plugins();

  geo_point p_ll{ loc1, crs_ll };
  geo_point p_utm{ loc3, crs_utm_18n };

  auto const conv_loc_utm = p_ll.location( p_utm.crs() );
  auto const conv_loc_ll = p_utm.location( p_ll.crs() );

  vector_3d _loc3{ loc3[0], loc3[1], 0 };
  auto const epsilon_ll_to_utm = ( _loc3 - conv_loc_utm ).norm();

  vector_3d _loc1{ loc1[0], loc1[1], 0 };
  auto const epsilon_utm_to_ll = ( _loc1 - conv_loc_ll ).norm();

  EXPECT_MATRIX_NEAR( p_ll.location(), conv_loc_ll, 1e-7 );
  EXPECT_MATRIX_NEAR( p_utm.location(), conv_loc_utm, 1e-2 );
  EXPECT_LT( epsilon_ll_to_utm, 1e-2 );
  EXPECT_LT( epsilon_utm_to_ll, 1e-7 );

  std::cout << "LL->UTM epsilon: " << epsilon_ll_to_utm << std::endl;
  std::cout << "UTM->LL epsilon: " << epsilon_utm_to_ll << std::endl;

  // Test with altitude
  geo_point p_lla{ loc1a, crs_ll };
  geo_point p_utma{ loc3a, crs_utm_18n };

  auto const conv_loc_utma = p_lla.location( p_utma.crs() );
  auto const conv_loc_lla = p_utma.location( p_lla.crs() );

  auto const epsilon_lla_to_utma = ( loc3a - conv_loc_utma ).norm();
  auto const epsilon_utma_to_lla = ( loc1a - conv_loc_lla ).norm();

  EXPECT_MATRIX_NEAR( p_lla.location(), conv_loc_lla, 1e-7 );
  EXPECT_MATRIX_NEAR( p_utma.location(), conv_loc_utma, 1e-2 );
  EXPECT_LT( epsilon_lla_to_utma, 1e-2 );
  EXPECT_LT( epsilon_utma_to_lla, 1e-7 );

  std::cout << "LLA->UTMa epsilon: " << epsilon_lla_to_utma << std::endl;
  std::cout << "UTMa->LLA epsilon: " << epsilon_utma_to_lla << std::endl;
}

// ----------------------------------------------------------------------------
TEST(geo_point, insert_operator_empty)
{
  kwiver::vital::geo_point p_empty;

  std::stringstream str;
  str << p_empty;

  EXPECT_EQ( "geo_point\n[ empty ]", str.str() );
}

// ----------------------------------------------------------------------------
struct roundtrip_test
{
  char const* text;
  geo_point point;
};

// ----------------------------------------------------------------------------
void
PrintTo( roundtrip_test const& v, ::std::ostream* os )
{
  (*os) << v.text;
}

// ----------------------------------------------------------------------------
class geo_point_roundtrip : public ::testing::TestWithParam<roundtrip_test>
{
};

// ----------------------------------------------------------------------------
TEST_P(geo_point_roundtrip, insert_operator)
{
  auto const p = GetParam().point;
  auto const expected_loc = p.location();
  auto const expected_crs = p.crs();

  // Write point to stream
  std::stringstream out;
  out << p;

  // Replace commas so we can read numbers back in
  auto s = out.str();
  std::replace( s.begin(), s.end(), ',', ' ' );

  // Read back values
  std::stringstream in(s);

  double easting, northing, altitude;
  int crs;
  std::string dummy;

  in >> dummy; // geo_point\n
  in >> dummy; // [
  in >> easting;
  in >> northing;
  in >> altitude;
  in >> dummy; // ]
  in >> dummy; // @
  in >> crs;

  // Successful round-trip?
  EXPECT_EQ( expected_loc[0], easting );
  EXPECT_EQ( expected_loc[1], northing );
  EXPECT_EQ( expected_loc[2], altitude );
  EXPECT_EQ( expected_crs, crs );
}

// ----------------------------------------------------------------------------
INSTANTIATE_TEST_CASE_P(
  ,
  geo_point_roundtrip,
  ::testing::Values(
      ( roundtrip_test{ "1", geo_point{ loc1, crs_ll } } ),
      ( roundtrip_test{ "2", geo_point{ loc2, crs_ll } } ),
      ( roundtrip_test{ "3", geo_point{ loc3, crs_utm_18n } } ),
      ( roundtrip_test{ "1a", geo_point{ loc1a, crs_ll } } ),
      ( roundtrip_test{ "2a", geo_point{ loc2a, crs_ll } } ),
      ( roundtrip_test{ "3a", geo_point{ loc3a, crs_utm_18n } } ),
      ( roundtrip_test{ "rt", geo_point{ locrt, 12345 } } )
  ) );
