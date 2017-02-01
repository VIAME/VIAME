/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief core polygon class tests
 */

#include <test_common.h>

#include <vital/types/polygon.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

// ------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(default_constructor)
{
  kwiver::vital::polygon p;

  if ( p.num_vertices() != 0 )
  {
    TEST_ERROR("The default polygon is not empty");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(constructor_vec)
{
  std::vector< kwiver::vital::polygon::point_t > vec;

  //                                              X    Y
  vec.push_back( kwiver::vital::polygon::point_t( 10, 10 ) );
  vec.push_back( kwiver::vital::polygon::point_t( 10, 50 ) );
  vec.push_back( kwiver::vital::polygon::point_t( 50, 50 ) );
  vec.push_back( kwiver::vital::polygon::point_t( 30, 30 ) );

  kwiver::vital::polygon p( vec );
  if ( p.num_vertices() != 4 )
  {
    TEST_ERROR("The polygon has too few vertices");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(constructor_point)
{
  kwiver::vital::polygon p;

  //                                              X    Y
  p.push_back( kwiver::vital::polygon::point_t( 10, 10 ) );
  p.push_back( kwiver::vital::polygon::point_t( 10, 50 ) );
  p.push_back( kwiver::vital::polygon::point_t( 50, 50 ) );
  p.push_back( kwiver::vital::polygon::point_t( 30, 30 ) );

  if ( p.num_vertices() != 4 )
  {
    TEST_ERROR("The polygon has too few vertices");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(api)
{
  kwiver::vital::polygon p;

  //                                              X    Y
  p.push_back( kwiver::vital::polygon::point_t( 10, 10 ) );
  p.push_back( kwiver::vital::polygon::point_t( 10, 50 ) );
  p.push_back( kwiver::vital::polygon::point_t( 50, 50 ) );
  p.push_back( kwiver::vital::polygon::point_t( 50, 10 ) );

  if ( ! p.contains( 30, 30 ) )
  {
    TEST_ERROR("The polygon does contain (30,30)");
  }

  if ( p.contains( 70, 70 ) )
  {
    TEST_ERROR("The polygon does not contain (70,70)");
  }

  auto pt = p.at(1);
  if ( pt[0] != 10 || pt[1] != 50 )
  {
    TEST_ERROR("The polygon point 1 is not correct" );
  }

  auto vec = p.get_vertices();

  // print vector
  const size_t limit = vec.size();
  for ( size_t i = 0; i < limit; ++i )
  {
    std::cout << "[" << vec[i][0] << ", " << vec[i][1] << "]" << std::endl;
  } // end for
}
