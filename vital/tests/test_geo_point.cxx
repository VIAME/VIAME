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
 * \brief core geo_point class tests
 */

#include <test_eigen.h>
#include <test_gtest.h>

#include <vital/types/geo_point.h>
#include <vital/types/geodesy.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <gtest/gtest.h>


static auto const loc1 = kwiver::vital::vector_2d{ -73.759291, 42.849631 };
static auto const loc2 = kwiver::vital::vector_2d{ -73.757161, 42.849764 };
static auto const loc3 = kwiver::vital::vector_2d{ 601375.01, 4744863.31 };

static auto constexpr crs_ll = kwiver::vital::SRID::lat_lon_WGS84;
static auto constexpr crs_utm_18n = kwiver::vital::SRID::UTM_WGS84_north + 18;

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );
  TEST_LOAD_PLUGINS();
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(geo_point, default_constructor)
{
  kwiver::vital::geo_point p;
  EXPECT_TRUE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_point, constructor_point)
{
  kwiver::vital::geo_point p{ loc1, crs_ll };
  EXPECT_FALSE( p.is_empty() );
}

// ----------------------------------------------------------------------------
TEST(geo_point, assignment)
{
  kwiver::vital::geo_point p;
  kwiver::vital::geo_point const p1{ loc1, crs_ll };
  kwiver::vital::geo_point const p2;

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
  kwiver::vital::geo_point p{ loc1, crs_ll };

  // Test values of the point as originally constructed
  EXPECT_EQ( crs_ll, p.crs() );
  EXPECT_EQ( loc1, p.location() );
  EXPECT_EQ( loc1, p.location( crs_ll ) );

  // Modify the location and test the new values
  p.set_location( loc3, crs_utm_18n );

  EXPECT_EQ( crs_utm_18n, p.crs() );
  EXPECT_EQ( loc3, p.location() );
  EXPECT_EQ( loc3, p.location( crs_utm_18n ) );

  // Modify the location again and test the new values
  p.set_location( loc2, crs_ll );

  EXPECT_EQ( crs_ll, p.crs() );
  EXPECT_EQ( loc2, p.location() );
  EXPECT_EQ( loc2, p.location( crs_ll ) );

  // Test that the old location is not cached
  try
  {
    EXPECT_NE( loc3, p.location( crs_utm_18n ) )
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
  kwiver::vital::geo_point p_ll{ loc1, crs_ll };
  kwiver::vital::geo_point p_utm{ loc3, crs_utm_18n };

  auto const conv_loc_utm = p_ll.location( p_utm.crs() );
  auto const conv_loc_ll = p_utm.location( p_ll.crs() );

  auto const epsilon_ll_to_utm = ( loc3 - conv_loc_utm ).norm();
  auto const epsilon_utm_to_ll = ( loc1 - conv_loc_ll ).norm();

  EXPECT_MATRIX_NEAR( p_ll.location(), conv_loc_ll, 1e-7 );
  EXPECT_MATRIX_NEAR( p_utm.location(), conv_loc_utm, 1e-2 );
  EXPECT_LT( epsilon_ll_to_utm, 1e-2 );
  EXPECT_LT( epsilon_utm_to_ll, 1e-7 );

  std::cout << "LL->UTM epsilon: " << epsilon_ll_to_utm << std::endl;
  std::cout << "UTM->LL epsilon: " << epsilon_utm_to_ll << std::endl;
}
