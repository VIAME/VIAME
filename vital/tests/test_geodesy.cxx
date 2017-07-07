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
 * \brief core geodesy tests
 */

#include <vital/types/geodesy.h>

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
    kwiver::vital::utm_ups_zone( { 0.0, -100.0 } ),
    std::range_error );
  EXPECT_THROW(
    kwiver::vital::utm_ups_zone( { 0.0, +100.0 } ),
    std::range_error );
}
