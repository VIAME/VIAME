/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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
