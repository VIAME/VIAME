// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief core image class tests
 */

#include <vital/types/timestamp.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(timestamp, api)
{
  kwiver::vital::timestamp ts;

  EXPECT_FALSE( ts.is_valid() );
  EXPECT_FALSE( ts.has_valid_time() );
  EXPECT_FALSE( ts.has_valid_frame() );

  kwiver::vital::timestamp tsv( 5000000, 2);

  EXPECT_TRUE( tsv.is_valid() );
  EXPECT_TRUE( tsv.has_valid_time() );
  EXPECT_TRUE( tsv.has_valid_frame() );
  EXPECT_EQ( 5000000, tsv.get_time_usec() );
  EXPECT_EQ( 5, tsv.get_time_seconds() );
  EXPECT_EQ( 2, tsv.get_frame() );

  kwiver::vital::timestamp ts_copy{ tsv };
  EXPECT_EQ( tsv, ts_copy = kwiver::vital::timestamp( 5000000, 2 ) );
  EXPECT_EQ( tsv, ts_copy = tsv );

  ts.set_frame( 123 );
  EXPECT_EQ( 123, ts.get_frame() );

  ts.set_time_usec( 123000000 );
  EXPECT_EQ( 123, ts.get_time_seconds() );
  EXPECT_EQ( 123000000, ts.get_time_usec() );

  ts.set_time_seconds( 456 );
  EXPECT_EQ( 456, ts.get_time_seconds() );
  EXPECT_EQ( 456000000, ts.get_time_usec() );
}

// ----------------------------------------------------------------------------
TEST(timestamp, comparisons)
{
  kwiver::vital::timestamp tsii; // invalid TS
  kwiver::vital::timestamp   ts( 5000000, 123 );
  kwiver::vital::timestamp tsll( 4000000, 122 );
  kwiver::vital::timestamp tsel( 5000000, 122 );
  kwiver::vital::timestamp tseg( 5000000, 124 );
  kwiver::vital::timestamp tsgg( 6000000, 124 );

  // Compare to invalid
  EXPECT_FALSE( ts == tsii );
  EXPECT_TRUE(  ts != tsii );
  EXPECT_FALSE( ts < tsii );
  EXPECT_FALSE( ts > tsii );

  // Compare to self
  EXPECT_TRUE(  ts == ts );
  EXPECT_FALSE( ts != ts );
  EXPECT_FALSE( ts < ts );
  EXPECT_FALSE( ts > ts );

  // Compare to lesser time and frame number
  EXPECT_FALSE( ts == tsll );
  EXPECT_TRUE(  ts != tsll );
  EXPECT_FALSE( ts < tsll );
  EXPECT_TRUE(  ts > tsll );

  // Compare to lesser frame number
  EXPECT_FALSE( ts == tsel );
  EXPECT_TRUE(  ts != tsel );
  EXPECT_FALSE( ts < tsel );
  EXPECT_FALSE( ts > tsel ); // eh?

  // Compare to greater frame number
  EXPECT_FALSE( ts == tseg );
  EXPECT_TRUE(  ts != tseg );
  EXPECT_FALSE( ts < tseg ); // eh?
  EXPECT_FALSE( ts > tseg );

  // Compare to greater time and frame number
  EXPECT_FALSE( ts == tsgg );
  EXPECT_TRUE(  ts != tsgg );
  EXPECT_TRUE(  ts < tsgg );
  EXPECT_FALSE( ts > tsgg );

  // Compare to equal time with no frame number
  kwiver::vital::timestamp tsei;
  tsei.set_time_seconds( 5 );
  EXPECT_TRUE(  ts == tsei ); // eh?
  EXPECT_FALSE( ts != tsei ); // eh?
  EXPECT_FALSE( ts < tsei );
  EXPECT_FALSE( ts > tsei );

  // Compare to greater time with no frame number
  kwiver::vital::timestamp tsgi;
  tsgi.set_time_seconds( 6 );
  EXPECT_FALSE( ts == tsgi );
  EXPECT_TRUE(  ts != tsgi );
  EXPECT_TRUE(  ts < tsgi );
  EXPECT_FALSE( ts > tsgi );

  // Compare to lesser frame number with no time
  kwiver::vital::timestamp tsil;
  tsil.set_frame( 122 );
  EXPECT_FALSE( ts == tsil );
  EXPECT_TRUE(  ts != tsil );
  EXPECT_FALSE( ts < tsil );
  EXPECT_TRUE(  ts > tsil );

  // Compare time-only to frame-only
  EXPECT_FALSE( tsei == tsil );
  EXPECT_TRUE(  tsei != tsil );
  EXPECT_FALSE( tsei < tsil );
  EXPECT_FALSE( tsei > tsil );

  // Compare to different domain
  kwiver::vital::timestamp tsdd = ts;
  tsdd.set_time_domain_index( 1 );

  EXPECT_FALSE( ts == tsdd );
  EXPECT_TRUE(  ts != tsdd );
  EXPECT_FALSE( ts < tsdd );
  EXPECT_FALSE( ts > tsdd );
}
