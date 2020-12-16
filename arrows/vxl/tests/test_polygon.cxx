// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core polygon class tests
 */

#include <arrows/vxl/polygon.h>

#include <gtest/gtest.h>

#include <sstream>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(polygon, conversions)
{
  auto p = std::make_shared< kwiver::vital::polygon >();

  //                                              X   Y
  p->push_back( kwiver::vital::polygon::point_t( 10, 10 ) );
  p->push_back( kwiver::vital::polygon::point_t( 10, 50 ) );
  p->push_back( kwiver::vital::polygon::point_t( 50, 50 ) );
  p->push_back( kwiver::vital::polygon::point_t( 30, 30 ) );

  auto xpoly = kwiver::arrows::vxl::vital_to_vxl( p );
  ASSERT_EQ( 4, xpoly->num_vertices() );

  // Convert back to vital_polygon
  auto vpoly =  kwiver::arrows::vxl::vxl_to_vital( *xpoly.get() );
  ASSERT_EQ( 4, vpoly->num_vertices() );

  auto x_sheet = xpoly->operator[](0);
  auto v_vert = vpoly->get_vertices();

  for ( size_t i = 0; i < xpoly->num_vertices(); ++i )
  {
    EXPECT_EQ( v_vert[i](0), x_sheet[i].x() ) << "Vertex at index " << i;
    EXPECT_EQ( v_vert[i](1), x_sheet[i].y() ) << "Vertex at index " << i;
  }
}
