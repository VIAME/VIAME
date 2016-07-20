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
 * \brief test point_2 functionality
 */

#include <test_common.h>

#include <vital/types/bounding_box.h>

#define TEST_ARGS      ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


IMPLEMENT_TEST( construct_bbox_i )
{
  kwiver::vital::bounding_box_i::vector_type tl( 12, 23 );
  kwiver::vital::bounding_box_i::vector_type  br( 200, 223 );
  kwiver::vital::bounding_box_i bb1( tl, br );

  auto ul = bb1.upper_left();
  auto lr = bb1.lower_right();

  if ( ul != tl || lr != br )
  {
    TEST_ERROR("Coordinates of bounding box not initialized correctly");
  }
}


IMPLEMENT_TEST( construct_bbox_d )
{
  kwiver::vital::bounding_box_d::vector_type tl( 12, 23 );
  kwiver::vital::bounding_box_d bb1( tl, 111, 222 );

  auto ul = bb1.upper_left();
  auto lr = bb1.lower_right();

  kwiver::vital::bounding_box_d::vector_type br( 12 + 111, 23 + 222 );

  if ( ul != tl || lr != br )
  {
    TEST_ERROR( "Coordinates of bounding box not initialized correctly" );
  }
}


IMPLEMENT_TEST(translate_bbox_d)
{
  kwiver::vital::bounding_box_d::vector_type tl( 12, 23 );
  kwiver::vital::bounding_box_d::vector_type br( 200, 223 );
  kwiver::vital::bounding_box_d::vector_type tr( 20, 10 );
  kwiver::vital::bounding_box_d bb1( tl, br );

  kwiver::vital::translate( bb1, tr );

  auto ul = bb1.upper_left();
  auto lr = bb1.lower_right();

  if ( ul[0] != 32 || ul[1] != 33 )
  {
    TEST_ERROR("ul coordinates of box not translated as expected");
  }

  if ( lr[0] != 220 || lr[1] != 233 )
  {
    TEST_ERROR("lr coordinates of box not translated as expected");
  }
}


IMPLEMENT_TEST(intersection_bbox_d)
{
  kwiver::vital::bounding_box_d::vector_type tl( 12, 23 );
  kwiver::vital::bounding_box_d::vector_type br( 200, 223 );
  kwiver::vital::bounding_box_d::vector_type tr( 120, 110 );
  kwiver::vital::bounding_box_d bb1( tl, br );

  kwiver::vital::bounding_box_d bb2 = bb1;
  kwiver::vital::translate( bb2, tr );
  kwiver::vital::bounding_box_d bb3 = kwiver::vital::intersection( bb1, bb2 );

  auto ul = bb3.upper_left();
  auto lr = bb3.lower_right();

  if ( ul[0] != 132 || ul[1] != 133 )
  {
    TEST_ERROR("ul coordinates of box not as expected");
  }

  if ( lr[0] != 200 || lr[1] != 223 )
  {
    TEST_ERROR("lr coordinates of box not as expected");
  }
}
