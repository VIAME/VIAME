// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core triangle_scan_iterator class tests
 */

#include <arrows/core/triangle_scan_iterator.h>
#include <vital/types/vector.h>
#include <gtest/gtest.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace
{
const vector_2d pt1(0, 0);
const vector_2d pt2(10, 0);
const vector_2d pt3(5, 10);
}

// ----------------------------------------------------------------------------
TEST(triangle_scan_iterator, iterate)
{
  kwiver::arrows::core::triangle_scan_iterator iter(pt1, pt2, pt3);

  for (iter.reset(); iter.next(); )
  {
    int d = static_cast<int>(std::ceil(static_cast<double>(iter.scan_y()) / 2));
    EXPECT_EQ(0 + d, iter.start_x());
    EXPECT_EQ(10 - d, iter.end_x());
  }

}
