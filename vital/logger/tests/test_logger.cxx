/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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

#include <vital/logger/logger.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <functional>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

//
// Assume basic logger
//

// ----------------------------------------------------------------------------
TEST(logger, levels)
{
  logger_handle_t log2 = get_logger( "main.logger2" );

  log2->set_level( kwiver_logger::LEVEL_WARN );

  EXPECT_FALSE( IS_TRACE_ENABLED( log2 ) );
  EXPECT_FALSE( IS_DEBUG_ENABLED( log2 ) );
  EXPECT_FALSE( IS_INFO_ENABLED( log2 ) );
  EXPECT_TRUE( IS_WARN_ENABLED( log2 ) );
  EXPECT_TRUE( IS_ERROR_ENABLED( log2 ) );
  EXPECT_TRUE( IS_FATAL_ENABLED( log2 ) );

  EXPECT_EQ( "main.logger2", log2->get_name() );

  EXPECT_EQ( kwiver_logger::LEVEL_WARN, log2->get_level() );

  log2->set_level( kwiver_logger::LEVEL_DEBUG );

  EXPECT_FALSE( IS_TRACE_ENABLED( log2 ) );
  EXPECT_TRUE( IS_DEBUG_ENABLED( log2 ) );
  EXPECT_TRUE( IS_INFO_ENABLED( log2 ) );
  EXPECT_TRUE( IS_WARN_ENABLED( log2 ) );
  EXPECT_TRUE( IS_ERROR_ENABLED( log2 ) );
  EXPECT_TRUE( IS_FATAL_ENABLED( log2 ) );

  EXPECT_EQ( kwiver_logger::LEVEL_DEBUG, log2->get_level() );

  // Test to see if we get the same logger back
  // Note: some implementations may give back an identical pointer, but others
  // may not, so instead compare that their log level is synchronized
  auto log = get_logger( "main.logger2" );

  EXPECT_EQ( kwiver_logger::LEVEL_DEBUG, log->get_level() );

  log2->set_level( kwiver_logger::LEVEL_INFO );

  EXPECT_EQ( kwiver_logger::LEVEL_INFO, log2->get_level() );
  EXPECT_EQ( kwiver_logger::LEVEL_INFO, log->get_level() );
}

// ----------------------------------------------------------------------------
TEST(logger, factory)
{
  // Need to unset any logger factory specification so we will use the
  // default factory
#if defined _WIN32
  _putenv( "VITAL_LOGGER_FACTORY=" );
#else
  unsetenv( "VITAL_LOGGER_FACTORY" );
#endif

  logger_handle_t log2 = get_logger( "main.logger" );

  EXPECT_EQ( "default_logger factory", log2->get_factory_name() );
}

// ----------------------------------------------------------------------------
TEST(logger, output)
{
  logger_handle_t log2 = get_logger( "main.logger2" );

  log2->set_level( kwiver_logger::LEVEL_TRACE );

  LOG_DEBUG( log2, "Test message" );

  LOG_ASSERT( log2, true, "This should pass." );
  LOG_ASSERT( log2, false, "This should generate an ERROR message." );
}

//
// Need to test
//
// - loadable logger modules
// set env to bad module - get default logger
//  "default_logger factory"
// set env to test logger plugin - get that one.
//
// - logger output.
