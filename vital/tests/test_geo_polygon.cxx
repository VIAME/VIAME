/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief core geo_polygon class tests
 */

#include <test_eigen.h>

#include <vital/types/geo_polygon.h>
#include <vital/types/geodesy.h>
#include <vital/types/polygon.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>


// "It's a magical place." -- P.C.
static auto const loc_ll = kwiver::vital::vector_2d{ -149.484444, -17.619482 };
static auto const loc_utm = kwiver::vital::vector_2d{ 236363.98, 8050181.74 };

static auto const loc2_ll = kwiver::vital::vector_2d{ -73.759291, 42.849631 };

static auto constexpr crs_ll = kwiver::vital::SRID::lat_lon_WGS84;
static auto constexpr crs_utm_6s = kwiver::vital::SRID::UTM_WGS84_south + 6;

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

} } // end namespace

// ----------------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, default_constructor)
{
  kwiver::vital::geo_polygon p;
  EXPECT_TRUE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, constructor_polygon)
{
  kwiver::vital::geo_polygon p{ { loc_ll }, crs_ll };
  EXPECT_FALSE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_polygon, assignment)
{
  kwiver::vital::geo_polygon p;
  kwiver::vital::geo_polygon const p1{ { loc_ll }, crs_ll };
  kwiver::vital::geo_polygon const p2;

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
  kwiver::vital::geo_polygon p{ { loc_ll }, crs_ll };

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
  kwiver::vital::geo_polygon p_ll{ { loc_ll }, crs_ll };
  kwiver::vital::geo_polygon p_utm{ { loc_utm }, crs_utm_6s };

  auto const d1 =
    p_ll.polygon( p_utm.crs() ).at( 0 )  - p_utm.polygon().at( 0 );
  auto const d2 =
    p_utm.polygon( p_ll.crs() ).at( 0 ) - p_ll.polygon().at( 0 );

  auto const epsilon_ll_to_utm = d1.squaredNorm();
  auto const epsilon_utm_to_ll = d2.squaredNorm();

  EXPECT_LT( epsilon_ll_to_utm, 1e-4 );
  EXPECT_LT( epsilon_utm_to_ll, 1e-13 );

  std::cout << "LL->UTM epsilon: " << epsilon_ll_to_utm << std::endl;
  std::cout << "UTM->LL epsilon: " << epsilon_utm_to_ll << std::endl;
}
