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

#include <test_common.h>

#include <vital/types/geo_point.h>
#include <vital/types/geodesy.h>
#include <vital/plugin_loader/plugin_manager.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

static auto const loc1 = kwiver::vital::vector_2d{ -73.759291, 42.849631 };
static auto const loc2 = kwiver::vital::vector_2d{ -73.757161, 42.849764 };
static auto const loc3 = kwiver::vital::vector_2d{ 601375.01, 4744863.31 };

static auto constexpr crs_ll = kwiver::vital::SRID::lat_lon_WGS84;
static auto constexpr crs_utm_18n = kwiver::vital::SRID::UTM_WGS84_north + 18;

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);
  kwiver::vital::plugin_manager::instance().load_all_plugins();

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

// ----------------------------------------------------------------------------
IMPLEMENT_TEST(default_constructor)
{
  kwiver::vital::geo_point p;

  if ( ! p.is_empty() )
  {
    TEST_ERROR("The default point is not empty");
  }
}

// ----------------------------------------------------------------------------
IMPLEMENT_TEST(constructor_point)
{
  kwiver::vital::geo_point p{ loc1, crs_ll };

  if ( p.is_empty() )
  {
    TEST_ERROR("The constructed point is empty");
  }
}

// ----------------------------------------------------------------------------
IMPLEMENT_TEST(assignment)
{
  kwiver::vital::geo_point p;
  kwiver::vital::geo_point const p1{ loc1, crs_ll };
  kwiver::vital::geo_point const p2;

  if ( ! p.is_empty() )
  {
    TEST_ERROR("The default point is not empty");
  }

  p = p1;

  if ( p.is_empty() )
  {
    TEST_ERROR("The point is empty after assignment from non-empty point");
  }

  if ( p.location() != p1.location() )
  {
    TEST_ERROR("The point has the wrong location after assignment");
  }

  if ( p.crs() != p1.crs() )
  {
    TEST_ERROR("The point has the wrong CRS after assignment");
  }

  p = p2;

  if ( ! p.is_empty() )
  {
    TEST_ERROR("The point is not empty after assignment from empty point");
  }
}

// ----------------------------------------------------------------------------
IMPLEMENT_TEST(api)
{
  kwiver::vital::geo_point p{ loc1, crs_ll };

  // Test values of the point as originally constructed
  if ( p.location() != loc1 )
  {
    TEST_ERROR("The original location is incorrect");
  }

  if ( p.crs() != crs_ll )
  {
    TEST_ERROR("The original CRS is incorrect");
  }

  if ( p.location( crs_ll ) != loc1 )
  {
    TEST_ERROR("The location (requested in the original CRS) is incorrect");
  }

  // Modify the location
  p.set_location( loc3, crs_utm_18n );

  // Test the new values
  if ( p.location() != loc3 )
  {
    TEST_ERROR("The original location is incorrect");
  }

  if ( p.crs() != crs_utm_18n )
  {
    TEST_ERROR("The original CRS is incorrect");
  }

  if ( p.location( crs_utm_18n ) != loc3 )
  {
    TEST_ERROR("The location (requested in the original CRS) is incorrect");
  }

  // Modify the location again
  p.set_location( loc2, crs_ll );

  // Test the new values
  if ( p.location() != loc2 )
  {
    TEST_ERROR("The original location is incorrect");
  }

  if ( p.crs() != crs_ll )
  {
    TEST_ERROR("The original CRS is incorrect");
  }

  if ( p.location( crs_ll ) != loc2 )
  {
    TEST_ERROR("The location (requested in the original CRS) is incorrect");
  }

  // Test that the old location is not cached
  try
  {
    if ( p.location( crs_utm_18n ) == loc3 )
    {
      TEST_ERROR("Changing the location did not clear the location cache");
    }
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
IMPLEMENT_TEST(conversion)
{
  kwiver::vital::geo_point p_ll{ loc1, crs_ll };
  kwiver::vital::geo_point p_utm{ loc3, crs_utm_18n };

  auto const d1 = kwiver::vital::vector_2d{ p_ll.location( p_utm.crs() ) - p_utm.location() };
  auto const d2 = kwiver::vital::vector_2d{ p_utm.location( p_ll.crs() ) - p_ll.location() };

  auto const e1 = d1.squaredNorm();
  auto const e2 = d2.squaredNorm();

  auto str = []( kwiver::vital::vector_2d const& p ){
    return std::to_string( p.x() ) + ", " + std::to_string( p.y() );
  };

  if ( e1 > 1e-4 )
  {
    TEST_ERROR("Result of LL->UTM conversion exceeds tolerance");
    std::cout << "  expected: " << str( p_utm.location() ) << std::endl;
    std::cout << "  actual: " << str( p_ll.location( p_utm.crs() ) ) << std::endl;
  }

  if ( e2 > 1e-13 )
  {
    TEST_ERROR("Result of UTM->LL conversion exceeds tolerance");
    std::cout << "  expected: " << str( p_ll.location() ) << std::endl;
    std::cout << "  actual: " << str( p_utm.location( p_ll.crs() ) ) << std::endl;
  }

  std::cout << "LL->UTM epsilon: " << e1 << std::endl;
  std::cout << "UTM->LL epsilon: " << e2 << std::endl;
}
