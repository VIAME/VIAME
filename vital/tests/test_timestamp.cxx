/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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

#include <test_common.h>

#include <vital/types/timestamp.h>

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
IMPLEMENT_TEST(timestamp_API)
{
  kwiver::vital::timestamp ts;

  TEST_EQUAL( "Valid timestamp", ts.is_valid(), false );
  TEST_EQUAL( "Valid time", ts.has_valid_time(), false );
  TEST_EQUAL( "Valid frame", ts.has_valid_frame(), false );

  kwiver::vital::timestamp tsv( 5000000, 2);

  TEST_EQUAL( "Valid timestamp", tsv.is_valid(), true );
  TEST_EQUAL( "Valid time", tsv.has_valid_time(), true );
  TEST_EQUAL( "Valid frame", tsv.has_valid_frame(), true );
  TEST_EQUAL( "Correct time", tsv.get_time_usec(), 5000000 );
  TEST_EQUAL( "Correct time", tsv.get_time_seconds(), 5 );
  TEST_EQUAL( "Correct frame", tsv.get_frame(), 2 );

  // Test copy constructor
  ts = kwiver::vital::timestamp( 5000000, 2 );
  TEST_EQUAL( "Equal from temp", ts, tsv );

  ts = tsv;
  TEST_EQUAL( "Equal from other ts", ts, tsv );

  ts.set_frame( 123 );
  TEST_EQUAL( "Direct frame access", ts.get_frame(), 123 );

  ts.set_time_seconds( 123 );
  TEST_EQUAL( "Direct time access", ts.get_time_seconds(), 123 );
  TEST_EQUAL( "Direct time access", ts.get_time_usec(), 123000000 );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(timestamp_relop)
{
  kwiver::vital::timestamp tsiv; // invalid TS
  kwiver::vital::timestamp   ts( 5000000, 123 );
  kwiver::vital::timestamp ts11( 5000000, 122 );
  kwiver::vital::timestamp ts12( 5000000, 124 );
  kwiver::vital::timestamp ts21( 4000000, 122 );
  kwiver::vital::timestamp ts22( 6000000, 124 );

  TEST_EQUAL( "Invalid timestamp ==", (ts == tsiv), false );
  TEST_EQUAL( "Invalid timestamp !=", (ts != tsiv), true );
  TEST_EQUAL( "Invalid timestamp <", (ts < tsiv), false );
  TEST_EQUAL( "Invalid timestamp >", (ts > tsiv), false );

  TEST_EQUAL( "Same timestamp ==", (ts == ts), true );
  TEST_EQUAL( "Same timestamp !=", (ts != ts), false );
  TEST_EQUAL( "Same timestamp <", (ts < ts), false );
  TEST_EQUAL( "Same timestamp >", (ts > ts), false );

  TEST_EQUAL( "ts11 timestamp ==", (ts == ts11), false );
  TEST_EQUAL( "ts11 timestamp !=", (ts != ts11), true );
  TEST_EQUAL( "ts11 timestamp <", (ts < ts11), false );
  TEST_EQUAL( "ts11 timestamp >", (ts > ts11), false );

  TEST_EQUAL( "ts12 timestamp ==", (ts == ts12), false );
  TEST_EQUAL( "ts12 timestamp !=", (ts != ts12), true );
  TEST_EQUAL( "ts12 timestamp <", (ts < ts12), false );
  TEST_EQUAL( "ts12 timestamp >", (ts > ts12), false );

  TEST_EQUAL( "ts21 timestamp ==", (ts == ts21), false );
  TEST_EQUAL( "ts21 timestamp !=", (ts != ts21), true );
  TEST_EQUAL( "ts21 timestamp <", (ts < ts21), false );
  TEST_EQUAL( "ts21 timestamp >", (ts > ts21), true );

  TEST_EQUAL( "ts22 timestamp ==", (ts == ts22), false );
  TEST_EQUAL( "ts22 timestamp !=", (ts != ts22), true );
  TEST_EQUAL( "ts22 timestamp <", (ts < ts22), true );
  TEST_EQUAL( "ts22 timestamp >", (ts > ts22), false );

  kwiver::vital::timestamp ts10;
  ts10.set_time_seconds( 5 );

  TEST_EQUAL( "ts10 timestamp ==", (ts == ts10), true );
  TEST_EQUAL( "ts10 timestamp !=", (ts != ts10), false );
  TEST_EQUAL( "ts10 timestamp <", (ts < ts10), false );
  TEST_EQUAL( "ts10 timestamp >", (ts > ts10), false );

  ts10.set_time_seconds( 6 );

  TEST_EQUAL( "ts10 timestamp ==", (ts == ts10), false );
  TEST_EQUAL( "ts10 timestamp !=", (ts != ts10), true );
  TEST_EQUAL( "ts10 timestamp <", (ts < ts10), true );
  TEST_EQUAL( "ts10 timestamp >", (ts > ts10), false );

  kwiver::vital::timestamp ts01;
  ts01.set_frame( 121 );

  TEST_EQUAL( "ts01 frame only timestamp ==", (ts == ts01), false );
  TEST_EQUAL( "ts01 frame only timestamp !=", (ts != ts01), true );
  TEST_EQUAL( "ts01 frame only timestamp <", (ts < ts01), false );
  TEST_EQUAL( "ts01 frame only timestamp >", (ts > ts01), true );

  ts01.set_time_domain_index( 1 );

  TEST_EQUAL( "ts01 frame only timestamp ==", (ts == ts01), false );
  TEST_EQUAL( "ts01 frame only timestamp !=", (ts != ts01), true );
  TEST_EQUAL( "ts01 frame only timestamp <", (ts < ts01), false );
  TEST_EQUAL( "ts01 frame only timestamp >", (ts > ts01), false );

  TEST_EQUAL( "ts10:ts01 timestamp ==", (ts10 == ts01), false );
  TEST_EQUAL( "ts10:ts01 timestamp !=", (ts10 != ts01), true );
  TEST_EQUAL( "ts10:ts01 timestamp <", (ts10 < ts01), false );
  TEST_EQUAL( "ts10:ts01 timestamp >", (ts10 > ts01), false  );
}
