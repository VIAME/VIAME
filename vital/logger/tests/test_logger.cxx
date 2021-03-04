// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
