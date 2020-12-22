// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test range indirection
 */

#include <vital/range/indirect.h>

#include <gtest/gtest.h>

#include <vector>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(range_indirect, mutating)
{
  auto test_values = std::vector< int >{ 1, 2, 3, 4, 5 };

  for ( auto iter : test_values | range::indirect )
  {
    if ( *iter == 3 )
    {
      *iter = 42;
    }
  }

  EXPECT_EQ( 42, test_values[2] );
}

// ----------------------------------------------------------------------------
TEST(range_indirect, assign_iterator)
{
  auto const test_values = std::vector< int >{ 1, 2, 3, 4, 5 };

  auto out_iter = test_values.end();

  for ( auto iter : test_values | range::indirect )
  {
    if ( *iter == 3 )
    {
      out_iter = iter;
    }
  }

  ASSERT_NE( test_values.end(), out_iter );
  EXPECT_EQ( 3, *out_iter );
}
