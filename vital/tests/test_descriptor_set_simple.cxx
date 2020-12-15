// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Tests for simple_descriptor_set.
 */

#include <gtest/gtest.h>

#include <vital/types/descriptor_set.h>

// ----------------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
// Test construction of simple descriptor set.
TEST( descriptor_set_simple, construct_default )
{
  using namespace kwiver::vital;
  simple_descriptor_set ds;
}

// ----------------------------------------------------------------------------
// Test using range-based for-loop with empty dscriptor-set
TEST( descriptor_set_simple, range_based_loop_empty )
{
  using namespace kwiver::vital;

  // Range-based loop over empty set.
  simple_descriptor_set ds;
  int i = 0;
  for( descriptor_sptr const d : ds )
  {
    ++i;
  }
  EXPECT_EQ( i, 0 );
}

// ----------------------------------------------------------------------------
// Test using range-based for-loop with non-empty dscriptor-set
TEST( descriptor_set_simple, range_based_loop )
{
  using namespace kwiver::vital;

  // Make simple vector of descriptor_sptr
  std::vector< descriptor_sptr > dsptr_vec;
  for( int i=0; i < 3; ++i )
  {
    descriptor_fixed<int,2> *d = new descriptor_fixed<int,2>();
    d->raw_data()[0] = i;
    d->raw_data()[1] = i;
    dsptr_vec.push_back( descriptor_sptr( d ) );
  }

  // Range-based loop over empty set.
  simple_descriptor_set ds( dsptr_vec );
  size_t i = 0;
  for( descriptor_sptr const d : ds )
  {
    EXPECT_EQ( d->size(), 2 );
    EXPECT_EQ( d->as_double()[0], i );
    EXPECT_EQ( d->as_double()[1], i );
    ++i;
  }
  EXPECT_EQ( i, 3 );
}
