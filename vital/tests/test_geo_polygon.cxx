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

#include <test_common.h>

#include <vital/types/geo_polygon.h>
#include <vital/types/geodesy.h>
#include <vital/types/polygon.h>
#include <vital/plugin_loader/plugin_manager.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

// "It's a magical place." -- P.C.
static auto const loc_ll = kwiver::vital::vector_2d{ -149.484444, -17.619482 };
static auto const loc_utm = kwiver::vital::vector_2d{ 236363.98, 8050181.74 };

static auto constexpr crs_ll = kwiver::vital::SRID::lat_lon_WGS84;
static auto constexpr crs_utm_6s = kwiver::vital::SRID::UTM_WGS84_south + 6;

// ----------------------------------------------------------------------------
bool operator==( kwiver::vital::polygon const& a,
                 kwiver::vital::polygon const& b )
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
bool operator!=( kwiver::vital::polygon const& a,
                 kwiver::vital::polygon const& b )
{
  return !( a == b );
}

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
  kwiver::vital::geo_polygon p;

  if ( ! p.is_empty() )
  {
    TEST_ERROR("The default polygon is not empty");
  }
}

// ----------------------------------------------------------------------------
IMPLEMENT_TEST(constructor_point)
{
  kwiver::vital::geo_polygon p{ { loc_ll }, crs_ll };

  if ( p.is_empty() )
  {
    TEST_ERROR("The constructed polygon is empty");
  }
}

// ----------------------------------------------------------------------------
IMPLEMENT_TEST(assignment)
{
  kwiver::vital::geo_polygon p;
  kwiver::vital::geo_polygon const p1{ { loc_ll }, crs_ll };
  kwiver::vital::geo_polygon const p2;

  if ( ! p.is_empty() )
  {
    TEST_ERROR("The default polygon is not empty");
  }

  p = p1;

  if ( p.is_empty() )
  {
    TEST_ERROR("The polygon is empty after assignment from non-empty polygon");
  }

  if ( p.polygon() != p1.polygon() )
  {
    TEST_ERROR("The polygon has the wrong location after assignment");
  }

  if ( p.crs() != p1.crs() )
  {
    TEST_ERROR("The polygon has the wrong CRS after assignment");
  }

  p = p2;

  if ( ! p.is_empty() )
  {
    TEST_ERROR("The polygon is not empty after assignment from polygon point");
  }
}

// ----------------------------------------------------------------------------
IMPLEMENT_TEST(api)
{
  // TODO: Implement these tests
  // This test case should replicate the geo_point api test case, but doing it
  // well with the current test framework is awkward... it would be much easier
  // with Google Test!
}

// ----------------------------------------------------------------------------
IMPLEMENT_TEST(conversion)
{
  kwiver::vital::geo_polygon p_ll{ { loc_ll }, crs_ll };
  kwiver::vital::geo_polygon p_utm{ { loc_utm }, crs_utm_6s };

  auto const d1 =
    p_ll.polygon( p_utm.crs() ).at( 0 )  - p_utm.polygon().at( 0 );
  auto const d2 =
    p_utm.polygon( p_ll.crs() ).at( 0 ) - p_ll.polygon().at( 0 );

  auto const e1 = d1.squaredNorm();
  auto const e2 = d2.squaredNorm();

  if ( e1 > 1e-4 )
  {
    TEST_ERROR("Result of LL->UTM conversion exceeds tolerance");
  }

  if ( e2 > 1e-13 )
  {
    TEST_ERROR("Result of UTM->LL conversion exceeds tolerance");
  }

  std::cout << "LL->UTM epsilon: " << e1 << std::endl;
  std::cout << "UTM->LL epsilon: " << e2 << std::endl;
}
