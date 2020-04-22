/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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

#include <vital/types/geodesy.h>
#include <vital/types/local_cartesian.h>

#include <vital/plugin_loader/plugin_manager.h>

static auto constexpr wgs = kwiver::vital::SRID::lat_lon_WGS84;
static auto constexpr crs_utm_18n = kwiver::vital::SRID::UTM_WGS84_north + 18;

static auto const origA = kwiver::vital::geo_point(kwiver::vital::vector_3d( -73.759291, 42.849631, 0 ), wgs);
static auto const origB = kwiver::vital::geo_point(kwiver::vital::vector_3d( 601375.01, 4744863.31, 0 ), crs_utm_18n);
static auto const offset1 = kwiver::vital::vector_3d{ 25, 55, 0 };
static auto const offset2 = kwiver::vital::vector_3d{ 250, 5500, 50 };
static auto const geo1 = kwiver::vital::geo_point(kwiver::vital::vector_3d( -73.75898515, 42.85012609, 0 ), wgs);
static auto const geo2 = kwiver::vital::geo_point(kwiver::vital::vector_3d( -73.75623008, 42.89913984, 52.381 ), wgs);

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(local_cartesian, constructor)
{
  kwiver::vital::local_cartesian lc1(origA);
  EXPECT_EQ(lc1.get_origin().location(), origA.location());
  EXPECT_EQ(lc1.get_orientation(), 0);

  kwiver::vital::local_cartesian lc2(origA,33);
  EXPECT_EQ(lc2.get_origin().location(), origA.location());
  EXPECT_EQ(lc2.get_orientation(), 33);
}

// ----------------------------------------------------------------------------
TEST(local_cartesian, api)
{
  kwiver::vital::local_cartesian lc1(origA);
  // Test values of the point as originally constructed
  EXPECT_EQ(lc1.get_origin().location(), origA.location());
  EXPECT_EQ(lc1.get_orientation(), 0);

  // Modify the location and test the new values
  lc1.set_origin(geo1, 33);
  EXPECT_EQ(lc1.get_origin().location(), geo1.location());
  EXPECT_EQ(lc1.get_orientation(), 33);

  // Modify the location again and test the new values
  lc1.set_origin(geo2, 22 );
  EXPECT_EQ(lc1.get_origin().location(), geo2.location());
  EXPECT_EQ(lc1.get_orientation(), 22);
}

void CompareLLA(kwiver::vital::vector_3d gp1, kwiver::vital::vector_3d gp2)
{
  EXPECT_NEAR(gp1[0], gp2[0], 1e-7);
  EXPECT_NEAR(gp1[1], gp2[1], 1e-7);
  EXPECT_NEAR(gp1[2], gp2[2], 1e-3);
}

// ----------------------------------------------------------------------------
TEST(local_cartesian, conversion)
{
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  kwiver::vital::geo_point geo_outA;
  kwiver::vital::vector_3d cart_outA;
  kwiver::vital::local_cartesian lc_lla(origA);

  // Get the geopoint from an offset
  lc_lla.convert_from_cartesian(offset1, geo_outA);
  CompareLLA(geo_outA.location(), geo1.location());

  // Now get the cartesian value from that geo_point
  lc_lla.convert_to_cartesian(geo_outA, cart_outA);
  CompareLLA(cart_outA, offset1);

  // Get the geopoint from another offset
  lc_lla.convert_from_cartesian(offset2, geo_outA);
  CompareLLA(geo_outA.location(), geo2.location());

  // Now get the cartesian value from that geo_point
  lc_lla.convert_to_cartesian(geo_outA, cart_outA);
  CompareLLA(cart_outA, offset2);

  // Convert with a origin based in UTM
  kwiver::vital::geo_point geo_outB;
  kwiver::vital::vector_3d cart_outB;
  kwiver::vital::local_cartesian lc_utm(origB);

  // Get the geopoint from an offset
  lc_utm.convert_from_cartesian(offset1, geo_outB);
  CompareLLA(geo_outB.location(), geo1.location());

  // Now get the cartesian value from that geo_point
  lc_utm.convert_to_cartesian(geo_outB, cart_outB);
  CompareLLA(cart_outB, offset1);

  // Get the geopoint from another offset
  lc_utm.convert_from_cartesian(offset2, geo_outB);
  CompareLLA(geo_outB.location(), geo2.location());

  // Now get the cartesian value from that geo_point
  lc_utm.convert_to_cartesian(geo_outB, cart_outB);
  CompareLLA(cart_outB, offset2);
}
