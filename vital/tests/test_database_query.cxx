// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test database_query class
 */

#include <vital/types/database_query.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

namespace {

std::vector<unsigned> const positive_samples  = { 2, 5, 6, 7, 8 };
std::vector<unsigned> const negative_samples  = { 1, 3, 4 };

}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(database_query, ensure_values)
{
  database_query::query_type qt = database_query::RETRIEVAL;

  database_query query;
  query.set_type( qt );

  EXPECT_EQ( query.type(), database_query::RETRIEVAL );
}

// ----------------------------------------------------------------------------
TEST(database_query, check_temporal_bounds)
{
  database_query query;
  timestamp ts_lower( 5000000, 123 );
  timestamp ts_upper( 6000000, 124 );

  // Set the bounds
  query.set_temporal_bounds(ts_lower, ts_upper);
  // Check that they were set
  EXPECT_EQ( query.temporal_lower_bound(), ts_lower );
  EXPECT_EQ( query.temporal_upper_bound(), ts_upper );
}

// ----------------------------------------------------------------------------
TEST(database_query, check_temporal_bounds_logic_error)
{
  database_query query;
  timestamp lower1( 3000000, 121 );
  timestamp upper1( 4000000, 122 );
  timestamp lower2( 5000000, 123 );
  timestamp upper2( 6000000, 124 );

  // Set it to some valid timestamps first
  query.set_temporal_bounds(lower1, upper1);
  EXPECT_EQ( query.temporal_lower_bound(), lower1 );
  EXPECT_EQ( query.temporal_upper_bound(), upper1 );

  // Now try adding improper bounds. where upper < lower
  // Should throw an std::logic_error
  ASSERT_THROW(query.set_temporal_bounds(upper2, lower2), std::logic_error);

  // Check that the previous bounds are still set
  EXPECT_EQ( query.temporal_lower_bound(), lower1 );
  EXPECT_EQ( query.temporal_upper_bound(), upper1 );

  // Check that one timestamp can be both bounds
  timestamp lower_and_upper( 700000, 125 );
  ASSERT_NO_THROW(query.set_temporal_bounds(lower_and_upper, lower_and_upper));

  // Check that the bounds were set correctly
  EXPECT_EQ( query.temporal_lower_bound(), lower_and_upper );
  EXPECT_EQ( query.temporal_upper_bound(), lower_and_upper );
}
